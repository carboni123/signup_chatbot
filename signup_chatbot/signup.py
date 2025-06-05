# signup_chatbot/signup.py
import os
import logging
import json
from typing import Optional, Tuple, Callable, Type, Dict, Any, List

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo
from llm_factory_toolkit import LLMClient, ToolFactory
from llm_factory_toolkit.exceptions import ProviderError, ToolError, LLMToolkitError

from .memory import Memory
from .tools.edit_user_profile import EditUserProfileTool
from .config_models import SignupConfig

module_logger = logging.getLogger(__name__)
# Basic logging configuration if no handlers are configured by the importing application
if not module_logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Signup:
    def __init__(
        self,
        user_model_cls: Type[BaseModel],
        get_user_profile_func: Callable[[str], Tuple[Optional[Dict[str, Any]], Optional[str]]],
        update_user_profile_func: Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]],
        signup_config: Optional[SignupConfig] = None,
        edit_tool_name: str = "edit_user_profile",
        edit_tool_description: str = "Updates the user's profile information based on provided details.",
        user_model_fields_to_ignore: Optional[List[str]] = None,
    ):
        """
        Initializes the Signup handler.

        Args:
            user_model_cls: The Pydantic model class representing the user's profile.
            get_user_profile_func: Callable (user_id) -> (profile_dict, error_msg)
                                   to retrieve the user's current profile.
            update_user_profile_func: Callable (user_id, update_dict) -> (success_bool, error_msg)
                                      to persist profile updates.
            signup_config: Configuration for the signup process (API keys, prompts, etc.).
            edit_tool_name: Custom name for the profile editing tool.
            edit_tool_description: Custom description for the profile editing tool.
        """
        self.user_model_cls = user_model_cls
        self.get_user_profile = get_user_profile_func
        self.update_user_profile = update_user_profile_func
        self.user_model_fields_to_ignore = user_model_fields_to_ignore or []
        self._llm_visible_profile_model_cls = self._create_llm_visible_model_type()

        self.config = signup_config or SignupConfig()

        self.sessions = Memory()

        self.tool_factory = ToolFactory()
        self.edit_user_profile_tool = EditUserProfileTool(
            update_user_state_func=self._internal_profile_update_wrapper,
            editable_fields_model=self._llm_visible_profile_model_cls,
            tool_name=edit_tool_name,
            tool_description=edit_tool_description,
        )
        self.tool_factory.register_tool(
            function=self.edit_user_profile_tool, 
            name=self.edit_user_profile_tool.name,
            description=self.edit_user_profile_tool.description, 
            parameters=self.edit_user_profile_tool.parameters
        )

        self.llm_client = LLMClient(
            provider_type="openai",
            api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY"),
            tool_factory=self.tool_factory,
            model=self.config.openai_default_model # Pass default model to provider
        )

        if not (self.config.openai_api_key or os.getenv("OPENAI_API_KEY")):
            module_logger.warning("OpenAI API key is not set. LLM calls will fail.")

    def _internal_profile_update_wrapper(
        self,
        user_id: str,
        profile_update: Dict[str, Any]
    ) -> bool:
        """Wrapper for the main app's update function to match tool's expectation."""
        success, error_msg = self.update_user_profile(user_id, profile_update)
        if not success:
            module_logger.error(f"Profile update failed for user {user_id} via wrapper: {error_msg}")
        return success

    def _create_llm_visible_model_type(self) -> Type[BaseModel]:
        """
        Dynamically creates a Pydantic model class that includes only the fields
        from self.user_model_cls that are NOT in self.fields_to_exclude_from_llm_prompt.
        """
        original_model_cls = self.user_model_cls

        fields_for_dynamic_model: Dict[str, Tuple[Any, FieldInfo]] = {}
        for name, field_info in original_model_cls.model_fields.items():
            if name not in self.user_model_fields_to_ignore:
                # We pass the original annotation and the full FieldInfo object
                # This preserves defaults, validators, descriptions, etc., for the included fields.
                fields_for_dynamic_model[name] = (field_info.annotation, field_info)

        # Sanitize model name for create_model
        dynamic_model_name = f"LLMVisible{original_model_cls.__name__}"
        dynamic_model_name = "".join(c if c.isalnum() else "_" for c in dynamic_model_name)

        # Attempt to carry over the model config
        # Pydantic's create_model uses a __config__ dict, not a ModelConfig class directly
        config_dict_for_dynamic_model = None
        if hasattr(original_model_cls, "model_config") and isinstance(original_model_cls.model_config, dict):
            # Pydantic v2 ConfigDict
            config_dict_for_dynamic_model = original_model_cls.model_config.copy()

        dynamic_model = create_model(
            dynamic_model_name, **fields_for_dynamic_model, __config__=config_dict_for_dynamic_model
        )
        module_logger.debug(
            f"Created dynamic model '{dynamic_model.__name__}' with fields: "
            f"{list(dynamic_model.model_fields.keys())}"
        )
        return dynamic_model

    def _get_user_profile_as_model(self, user_id: str) -> Tuple[Optional[BaseModel], Optional[str]]:
        profile_dict, error_msg = self.get_user_profile(user_id) # This is from the app (e.g., examples/interactive_test.py)

        if error_msg: # If the app's get_user_profile function itself reported an error
            module_logger.error(f"Error from get_user_profile_func for {user_id}: {error_msg}")
            return None, error_msg

        # If profile_dict is None (meaning new user or no data), initialize data_for_model with user_id.
        # Otherwise, use the retrieved profile_dict.
        if profile_dict is None:
            data_for_model = {"user_id": user_id} # Ensure user_id is present for new profiles
        else:
            data_for_model = profile_dict.copy() # Use a copy
            if "user_id" not in data_for_model: # Ensure user_id is in the dict from DB
                data_for_model["user_id"] = user_id

        try:
            # Filter to only known fields to avoid validation errors for extra db fields
            # not defined in self.user_model_cls, but ensure user_id is always passed if it's part of the model.
            valid_data = {k: v for k, v in data_for_model.items() if k in self.user_model_cls.model_fields}

            # Ensure 'user_id' is in valid_data if it's a field in the model,
            # especially if it might have been filtered out or was missing from profile_dict.
            if "user_id" in self.user_model_cls.model_fields and "user_id" not in valid_data:
                valid_data["user_id"] = user_id # Add it from the method argument

            user_profile_instance = self.user_model_cls(**valid_data)
            return user_profile_instance, None
        except ValidationError as e:
            module_logger.error(f"Validation error parsing profile for {user_id} into {self.user_model_cls.__name__}. Data: {valid_data}, Error: {e}")
            return None, f"Error parsing user profile data: {e}"
        except Exception as e: # Catch other potential errors during model instantiation
            module_logger.error(f"Unexpected error instantiating {self.user_model_cls.__name__} for {user_id} with data {valid_data}: {e}", exc_info=True)
            return None, f"Internal error processing user profile: {e}"

    def get_missing_fields(self, user_profile_instance: BaseModel) -> List[str]:
        missing = []
        for field_name in self.user_model_cls.model_fields.keys():
            # Ignore fields
            if field_name in self.user_model_fields_to_ignore:
                continue
            value = getattr(user_profile_instance, field_name, None)
            if value is None: # Considers a field missing if its value is None
                missing.append(field_name)
        return missing

    def is_signup_complete(self, user_id: str) -> bool:
        if self.sessions.get_session_metadata_value(user_id, "signup_complete", False):
            return True

        user_profile_instance, error_msg = self._get_user_profile_as_model(user_id)
        if error_msg or user_profile_instance is None:
            module_logger.warning(
                f"Cannot determine signup completion for {user_id} due to profile error: {error_msg}"
            )
            return False
        return not self.get_missing_fields(user_profile_instance)

    async def handle_message(self, user_id: str, user_input: str, interaction_context: str = "new_user") -> List[str]:
        module_logger.info(f"Handling message for user_id: {user_id}, input: '{user_input[:100]}...'")

        final_bot_messages: List[str] = []

        # Check history BEFORE adding current user_input to determine if this is the first real exchange.
        history_before_this_turn = self.sessions.get_messages(user_id)
        is_first_exchange = not history_before_this_turn

        user_profile_instance, profile_error = self._get_user_profile_as_model(user_id)

        # Handle profile loading errors early
        if profile_error:
            module_logger.error(f"Profile error for {user_id}: {profile_error}")
            error_msg = self.config.default_error_message
            # Save user's message first, then the error, before returning
            self.sessions.save_message(user_id, content=user_input, role="user")
            self.sessions.save_message(user_id, content=error_msg, role="assistant")
            return [error_msg]
        if user_profile_instance is None: # Should be caught by profile_error, but as a safeguard
            module_logger.critical(f"User profile instance is None for {user_id} without profile_error. This is unexpected.")
            error_msg = self.config.default_error_message
            self.sessions.save_message(user_id, content=user_input, role="user")
            self.sessions.save_message(user_id, content=error_msg, role="assistant")
            return [error_msg]

        # Determine if signup has been marked complete via skip command
        signup_completed_flag = self.is_signup_complete(user_id)

        # Check for skip command on this turn
        if (
            not signup_completed_flag
            and self.config.skip_command
            and user_input.strip().lower() == self.config.skip_command.lower()
        ):
            self.sessions.update_session_metadata(user_id, "signup_complete", True)
            self.sessions.save_message(user_id, content=user_input, role="user")
            completion_msg = self.config.signup_complete_message
            final_bot_messages.append(completion_msg)
            self.sessions.save_message(user_id, content=completion_msg, role="assistant")
            module_logger.info(f"User {user_id} invoked skip command; signup marked complete.")
            return final_bot_messages

        missing_fields = [] if signup_completed_flag else self.get_missing_fields(user_profile_instance)
        signup_is_incomplete = bool(missing_fields)

        # Save first so the LLM knows the context
        self.sessions.save_message(user_id, content=user_input, role="user")

        # Add initial welcome message if it's the first exchange and signup is incomplete
        if is_first_exchange and signup_is_incomplete and interaction_context != "new_user":
            welcome_message = self.config.welcome_back_incomplete
            final_bot_messages.append(welcome_message)
            self.sessions.save_message(user_id, content=welcome_message, role="assistant")
            module_logger.info(f"Added initial welcome message for {user_id}: '{welcome_message}'")
        elif is_first_exchange and signup_is_incomplete:
            welcome_message = self.config.welcome_message
            final_bot_messages.append(welcome_message)
            self.sessions.save_message(user_id, content=welcome_message, role="assistant")
            module_logger.info(f"Added initial welcome message for {user_id}: '{welcome_message}'")

        # Prepare system prompt with current profile state
        # missing_fields here are based on user_profile_instance fetched *before* this turn's LLM interaction.
        profile_json_for_prompt = user_profile_instance.model_dump_json(indent=2, exclude_none=True)
        system_prompt_content = self.config.llm_system_prompt_template.format(
            user_profile_json=profile_json_for_prompt,
            missing_fields_list=", ".join(missing_fields) if missing_fields else "All fields were filled. Signup completed."
        )

        messages_for_llm: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_content},
        ]
        chat_history = self.sessions.get_messages(user_id) # Includes current user message
        messages_for_llm.extend(chat_history[-self.config.history_limit:])

        llm_response_text = ""
        try:
            module_logger.debug(f"LLMClient.generate call for {user_id}. Messages: {json.dumps(messages_for_llm, indent=2)}")
            llm_response_text, _ = await self.llm_client.generate(
                messages=messages_for_llm,
                use_tools=["edit_user_profile"],
                tool_execution_context={"user_id": user_id},
                temperature=0.5,
            )
        except (ProviderError, ToolError, LLMToolkitError) as e:
            module_logger.exception(f"LLM Toolkit error during interaction for {user_id}: {e}")
            llm_response_text = self.config.default_error_message
        except Exception as e:
            module_logger.exception(f"Unexpected error during LLM interaction for {user_id}: {e}")
            llm_response_text = self.config.default_error_message

        # Fallback logic for empty LLM response
        if not llm_response_text:
            # Re-fetch profile as LLM tool calls might have updated it
            profile_after_llm, profile_fetch_err = self._get_user_profile_as_model(user_id)
            if profile_fetch_err or not profile_after_llm:
                module_logger.error(f"Error fetching profile after LLM for fallback: {profile_fetch_err}")
                llm_response_text = self.config.default_error_message
            else:
                current_missing_fields_after_llm = self.get_missing_fields(profile_after_llm)
                if not current_missing_fields_after_llm: # Signup is now complete
                    llm_response_text = self.config.signup_complete_message
                else: # Signup still incomplete
                    # current_missing_fields_after_llm should not be empty if signup is incomplete
                    llm_response_text = self.config.signup_prompt_format.format(fields_list=current_missing_fields_after_llm[0])

        # Add LLM's response (or fallback) to the list and save to history
        if llm_response_text: # Ensure we don't add an empty string
            final_bot_messages.append(llm_response_text)
            self.sessions.save_message(user_id, content=llm_response_text, role="assistant")
        elif not final_bot_messages:
            # This case is reached if:
            # 1. No initial_welcome_message was added (i.e., not first exchange OR signup was already complete).
            # 2. llm_response_text is empty (LLM returned nothing, and fallback also resulted in empty string).
            # To prevent returning an empty list, add a generic fallback message.
            module_logger.warning(f"LLM and fallback produced no message for {user_id}, and no prior message in this turn's response list.")
            generic_fallback = "I'm sorry, I couldn't generate a response at this moment. Could you try rephrasing or asking again?"
            final_bot_messages.append(generic_fallback)
            self.sessions.save_message(user_id, content=generic_fallback, role="assistant")

        module_logger.info(f"Final bot replies for {user_id}: {final_bot_messages}")
        return final_bot_messages
