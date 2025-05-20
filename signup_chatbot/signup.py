# signup_chatbot/signup.py
import os
import logging
import json
from typing import Optional, Tuple, Callable, Type, Dict, Any, List

from pydantic import BaseModel, ValidationError
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
        
        self.config = signup_config or SignupConfig()

        self.sessions = Memory()
        
        self.tool_factory = ToolFactory()
        self.edit_user_profile_tool = EditUserProfileTool(
            update_user_state_func=self._internal_profile_update_wrapper,
            editable_fields_model=self.user_model_cls,
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
            value = getattr(user_profile_instance, field_name, None)
            if value is None: # Considers a field missing if its value is None
                missing.append(field_name)
        return missing

    def is_signup_complete(self, user_id: str) -> bool:
        user_profile_instance, error_msg = self._get_user_profile_as_model(user_id)
        if error_msg or user_profile_instance is None:
            module_logger.warning(f"Cannot determine signup completion for {user_id} due to profile error: {error_msg}")
            return False
        return not self.get_missing_fields(user_profile_instance)

    async def handle_message(self, user_id: str, user_input: str) -> str:
        module_logger.info(f"Handling message for user_id: {user_id}, input: '{user_input[:100]}...'") # Log truncated input

        self.sessions.save_message(user_id, content=user_input, role="user")

        user_profile_instance, profile_error = self._get_user_profile_as_model(user_id)
        if profile_error:
            module_logger.error(f"Profile error for {user_id}: {profile_error}")
            self.sessions.save_message(user_id, content=self.config.default_error_message, role="assistant")
            return self.config.default_error_message
        if user_profile_instance is None: # Should be caught by profile_error typically
             module_logger.critical(f"User profile instance is None for {user_id} without profile_error. This is unexpected.")
             self.sessions.save_message(user_id, content=self.config.default_error_message, role="assistant")
             return self.config.default_error_message

        missing_fields = self.get_missing_fields(user_profile_instance)
        module_logger.info(f"Current missing fields: {missing_fields}")

        # The system prompt will guide the LLM. If it's the first turn, LLM should naturally greet and ask.
        # If signup is complete, the prompt tells the LLM.
        profile_json_for_prompt = user_profile_instance.model_dump_json(indent=2, exclude_none=True)
        system_prompt_content = self.config.llm_system_prompt_template.format(
            user_profile_json=profile_json_for_prompt,
            missing_fields_list=", ".join(missing_fields) if missing_fields else "All fields filled."
        )

        messages_for_llm: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_content},
        ]
        chat_history = self.sessions.get_messages(user_id) # Includes current user message
        messages_for_llm.extend(chat_history[-self.config.history_limit:])
        
        
        llm_response_text = ""
        try:
            module_logger.debug(f"LLMClient.generate call for {user_id}. Messages: {json.dumps(messages_for_llm, indent=2)}")
            
            # The LLMClient and its provider will handle the tool call loop
            llm_response_text, _ = await self.llm_client.generate(
                messages=messages_for_llm,
                # model=self.config.openai_default_model, # Already set in LLMClient init or can override here
                use_tools=["edit_user_profile"],
                tool_execution_context={"user_id": user_id}, # Pass user_id for the tool
                temperature=0.5,
                # max_tool_iterations can be passed if needed (default is 5 in OpenAIProvider)
            )

        except (ProviderError, ToolError, LLMToolkitError) as e: # Catch toolkit specific errors
            module_logger.exception(f"LLM Toolkit error during interaction for {user_id}: {e}")
            llm_response_text = self.config.default_error_message
        except Exception as e: # General errors
            module_logger.exception(f"Unexpected error during LLM interaction for {user_id}: {e}")
            llm_response_text = self.config.default_error_message

        # Fallback if LLM response is empty
        if not llm_response_text:
            if self.is_signup_complete(user_id):
                llm_response_text = self.config.signup_complete_message
            else:
                # Try to prompt for the next specific missing field as a simple fallback
                updated_profile_instance, _ = self._get_user_profile_as_model(user_id) # Re-fetch
                if updated_profile_instance:
                    current_missing_fields = self.get_missing_fields(updated_profile_instance)
                    if current_missing_fields:
                        llm_response_text = self.config.signup_prompt_format.format(fields_list=current_missing_fields[0])
                    else: # Signup just completed, LLM didn't say anything
                        llm_response_text = self.config.signup_complete_message
                else: # Should not happen
                    llm_response_text = self.config.default_error_message


        self.sessions.save_message(user_id, content=llm_response_text, role="assistant")
        module_logger.info(f"Final bot reply for {user_id}: {llm_response_text}")
        return llm_response_text