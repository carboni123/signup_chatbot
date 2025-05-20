# signup_chatbot/signup.py
import os
import logging
import json
from typing import Optional, Tuple, Callable, Type, Dict, Any, List

from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI, OpenAIError # Import OpenAIError for specific exception handling
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall # For type hinting

from .memory import Memory
from .tools.edit_user_profile import EditUserProfileTool
from .tools.tool_models import ToolExecutionResult
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
        self.client = AsyncOpenAI(
            api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        if not (self.config.openai_api_key or os.getenv("OPENAI_API_KEY")):
            module_logger.warning("OpenAI API key is not set. LLM calls will fail.")

        self.edit_user_profile_tool = EditUserProfileTool(
            update_user_state_func=self._internal_profile_update_wrapper,
            editable_fields_model=self.user_model_cls,
            tool_name=edit_tool_name,
            tool_description=edit_tool_description,
        )

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
        profile_dict, error_msg = self.get_user_profile(user_id)
        if error_msg:
            return None, error_msg
        
        data_for_model = profile_dict if profile_dict is not None else {}
        try:
            # Filter to only known fields to avoid validation errors for extra db fields
            valid_data = {k: v for k, v in data_for_model.items() if k in self.user_model_cls.model_fields}
            user_profile_instance = self.user_model_cls(**valid_data)
            return user_profile_instance, None
        except ValidationError as e:
            module_logger.error(f"Validation error parsing profile for {user_id} into {self.user_model_cls.__name__}: {e}")
            return None, f"Error parsing user profile data: {e}"
        except Exception as e:
            module_logger.error(f"Error instantiating {self.user_model_cls.__name__} for {user_id}: {e}")
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
        
        tool_definitions = [self.edit_user_profile_tool.get_openai_tool_definition()]
        
        llm_response_text = ""
        try:
            module_logger.debug(f"LLM call for {user_id}. Messages: {json.dumps(messages_for_llm, indent=2)}. Tools: {[t['function']['name'] for t in tool_definitions]}")
            response = await self.client.chat.completions.create(
                model=self.config.openai_default_model,
                messages=messages_for_llm, # type: ignore
                tools=tool_definitions, # type: ignore
                tool_choice="auto",
                temperature=0.5, # Consider making configurable
            )
            response_message: ChatCompletionMessage = response.choices[0].message
            
            current_llm_messages_for_follow_up = list(messages_for_llm)
            current_llm_messages_for_follow_up.append(response_message.model_dump(exclude_none=True))


            if response_message.tool_calls:
                module_logger.info(f"LLM requested tool calls for {user_id}: {response_message.tool_calls}")
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    tool_call_id = tool_call.id
                    
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        module_logger.error(f"Invalid JSON in tool arguments for {function_name}: {tool_call.function.arguments}. Error: {e}")
                        tool_response_str = json.dumps({"status": "error", "message": "Invalid arguments format."})
                    else:
                        module_logger.info(f"Executing tool '{function_name}' for {user_id} with args: {function_args}")
                        if function_name == self.edit_user_profile_tool.name:
                            tool_exec_result: ToolExecutionResult = self.edit_user_profile_tool(user_id=user_id, **function_args)
                            tool_response_str = tool_exec_result.content
                        else:
                            module_logger.warning(f"Unknown tool '{function_name}' requested by LLM for {user_id}.")
                            tool_response_str = json.dumps({"status": "error", "message": f"Tool '{function_name}' not found."})
                    
                    module_logger.info(f"Tool '{function_name}' response for {user_id}: {tool_response_str}")
                    current_llm_messages_for_follow_up.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_response_str,
                    })
                
                module_logger.info(f"Requesting LLM follow-up for {user_id} after tool execution(s).")
                follow_up_response = await self.client.chat.completions.create(
                    model=self.config.openai_default_model,
                    messages=current_llm_messages_for_follow_up, # type: ignore
                    temperature=0.5,
                )
                llm_response_text = follow_up_response.choices[0].message.content or ""
                if follow_up_response.choices[0].message.tool_calls:
                     module_logger.warning(f"LLM attempted a new tool call in follow-up for {user_id}. Ignoring. Content: '{llm_response_text}'")
                module_logger.info(f"LLM follow-up response for {user_id}: {llm_response_text}")

                # If LLM gives an empty response after a successful update, use default ack.
                if not llm_response_text:
                    # Check if profile was actually updated by parsing tool_response_str from the relevant tool call
                    # This part could be more robust by checking the status in tool_response_str
                    llm_response_text = self.config.profile_updated_ack

            else: # No tool calls
                llm_response_text = response_message.content or ""
                module_logger.info(f"LLM response (no tool call) for {user_id}: {llm_response_text}")

        except OpenAIError as e: # More specific OpenAI errors
            module_logger.exception(f"OpenAI API error during LLM call for {user_id}: {e}")
            llm_response_text = self.config.default_error_message
        except Exception as e:
            module_logger.exception(f"Unexpected error during LLM interaction for {user_id}: {e}")
            llm_response_text = self.config.default_error_message

        # Fallback if LLM response is empty
        if not llm_response_text.strip():
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