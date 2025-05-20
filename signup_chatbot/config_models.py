# signup_chatbot/config_models.py
from typing import Optional
from pydantic import BaseModel, Field

# Default prompts (English, can be overridden via SignupConfig)
DEFAULT_WELCOME_MESSAGE = "Welcome! To get started, I need a bit of information to set up your profile."
DEFAULT_SIGNUP_PROMPT_FORMAT = "Please provide your {fields_list}."
DEFAULT_LLM_SYSTEM_PROMPT_TEMPLATE = """Your primary goal is to complete the user's profile by collecting necessary information.
The user's profile structure is defined by a model, and you need to fill its fields.
Use the 'edit_user_profile' tool to save any information the user provides.
Ask for missing information conversationally, one piece or a small logical group at a time.
If the user provides multiple pieces of information at once, try to capture all of them.

Current User Profile (JSON format):
```json
{user_profile_json}
```

Fields that are currently missing or need to be collected: {missing_fields_list}

If all essential fields are complete, you can confirm this and ask if the user needs further assistance.
Respond in the language of the user's input if discernible, otherwise default to a common conversational language.
Be polite and helpful.
"""
DEFAULT_PROFILE_UPDATED_ACK = "Thanks, I've updated your profile with the information provided."
DEFAULT_ERROR_MESSAGE = "I'm sorry, but I encountered an issue. Please try again."
DEFAULT_SIGNUP_COMPLETE_MESSAGE = "Great, your profile setup is complete! How can I assist you further?"


class SignupConfig(BaseModel):
    openai_api_key: Optional[str] = Field(
        None, description="OpenAI API key. If None, attempts to use OPENAI_API_KEY env var."
    )
    openai_default_model: str = Field("gpt-4o-mini", description="Default OpenAI model to use for chat completions.")
    history_limit: int = Field(10, description="Number of past messages to include in the context for the LLM.")

    welcome_message: str = Field(
        DEFAULT_WELCOME_MESSAGE, description="Initial welcome message if needed (though LLM usually handles greeting)."
    )
    signup_prompt_format: str = Field(
        DEFAULT_SIGNUP_PROMPT_FORMAT,
        description="Format string for prompting missing fields, e.g., 'Please provide your {fields_list}'.",
    )
    llm_system_prompt_template: str = Field(
        DEFAULT_LLM_SYSTEM_PROMPT_TEMPLATE,
        description="System prompt template for the LLM. Use {user_profile_json} and {missing_fields_list}.",
    )
    profile_updated_ack: str = Field(
        DEFAULT_PROFILE_UPDATED_ACK, description="Acknowledgement message after profile is updated."
    )
    default_error_message: str = Field(DEFAULT_ERROR_MESSAGE, description="Default error message to show to the user.")
    signup_complete_message: str = Field(
        DEFAULT_SIGNUP_COMPLETE_MESSAGE, description="Message to indicate signup completion."
    )
