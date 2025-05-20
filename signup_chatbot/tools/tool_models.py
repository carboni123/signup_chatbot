# signup_chatbot/tools/tool_models.py
from typing import Optional, Any
from pydantic import BaseModel

class ToolExecutionResult(BaseModel):
    """
    Represents the result of a tool's execution.
    """
    content: str  # Typically a JSON string summarizing the outcome for the LLM
    action_needed: bool = False # Indicates if further action is required from the agent/LLM based on this tool's direct output
    error: Optional[str] = None # Detailed error message for logging/debugging, not usually for LLM
    
    # You can add other fields if necessary, e.g., raw_result: Any