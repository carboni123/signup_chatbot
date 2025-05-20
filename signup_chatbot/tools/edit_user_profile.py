# signup_chatbot/tools/edit_user_profile.py
import logging
import json
from typing import Dict, Any, Callable, Type, Optional

from pydantic import BaseModel, ValidationError
from .tool_models import ToolExecutionResult

module_logger = logging.getLogger(__name__)

class EditUserProfileTool:
    """
    Allows the LLM to request updates to specified fields of the 'User Profile'.
    The specific fields available for editing are determined by the 'editable_fields_model'
    provided during initialization.
    """
    name: str
    description: str
    parameters: Dict[str, Any] 

    _update_user_state_func: Callable[[str, Dict[str, Any]], bool]
    _editable_fields_model: Type[BaseModel]

    def __init__(
        self,
        update_user_state_func: Callable[[str, Dict[str, Any]], bool], 
        editable_fields_model: Type[BaseModel],
        tool_name: str = "edit_user_profile",
        tool_description: str = (
            "Updates the user's stored profile information. "
            "Use this when the user provides or wants to change their details required for signup or long-term preferences."
        ),
    ):
        self._update_user_state_func = update_user_state_func
        self._editable_fields_model = editable_fields_model
        self.name = tool_name
        self.description = tool_description
        
        schema = self._editable_fields_model.model_json_schema()
        self.parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }

    def get_openai_tool_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def __call__(
        self,
        user_id: str,
        **kwargs: Any
    ) -> ToolExecutionResult:
        module_logger.info(f"Executing {self.name} tool for user_id: {user_id} with raw args: {kwargs}")
        
        error_message: Optional[str] = None
        try:
            profile_update_model = self._editable_fields_model(**kwargs)
            profile_update = profile_update_model.model_dump(exclude_unset=True, by_alias=False)
        except ValidationError as e:
            error_message = f"Invalid parameters provided for {self.name}: {e.errors()}"
            module_logger.warning(f"{error_message} for user_id: {user_id}")
            result_dict = {"status": "error", "message": error_message}
            return ToolExecutionResult(
                content=json.dumps(result_dict),
                error=error_message
            )

        if not profile_update:
            module_logger.warning(f"Tool '{self.name}' called for {user_id} but no valid update parameters provided.")
            result_dict = {"status": "no_action", "message": "No profile update parameters were specified or valid."}
            return ToolExecutionResult(content=json.dumps(result_dict))

        module_logger.debug(f"Validated profile update for user_id {user_id}: {profile_update}")

        try:
            success = self._update_user_state_func(
                user_id=user_id,
                profile_update=profile_update
            )

            if success:
                updated_fields_str = ", ".join(profile_update.keys())
                success_message = f"Successfully updated user profile. Updated fields: {updated_fields_str}."
                module_logger.debug(f"{success_message} for user_id: {user_id}")
                result_dict = {"status": "success", "message": success_message, "updated_fields": list(profile_update.keys())}
            else:
                error_message = f"Update function returned False for user_id: {user_id}, update: {profile_update}"
                module_logger.error(error_message)
                result_dict = {"status": "error", "message": error_message}

        except Exception as e:
            error_message = f"An unexpected error occurred while updating user profile for {user_id}: {e}"
            module_logger.exception(f"Unexpected error in tool '{self.name}': {error_message}") # Logs full traceback
            result_dict = {"status": "error", "message": error_message}

        return ToolExecutionResult(
            content=json.dumps(result_dict),
            error=error_message
        )