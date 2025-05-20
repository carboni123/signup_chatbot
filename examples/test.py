# examples/test.py
import asyncio
import os
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signup_chatbot import Signup, SignupConfig

# 1. Define Your User Profile Model:
class MyUserProfile(BaseModel):
    name: Optional[str] = Field(None, description="User's full name")
    email: Optional[EmailStr] = Field(None, description="User's email address")
    birth_year: Optional[int] = Field(None, description="User's year of birth, e.g., 1990", ge=1900, le=2024) # Adjusted le
    

# 2. Implement Profile Get/Update Functions:
# Example using a simple dictionary as a mock DB
USER_DB: Dict[str, Dict[str, Any]] = {}

def get_my_user_profile(user_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # print(f"DEBUG: get_my_user_profile called for {user_id}. DB: {USER_DB}")
    if user_id in USER_DB:
        return USER_DB[user_id].copy(), None # Return a copy to avoid direct modification
    return None, None # Or {}, None if you prefer to return an empty dict for new users

def update_my_user_profile(user_id: str, profile_update: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # print(f"DEBUG: update_my_user_profile called for {user_id} with {profile_update}")
    if user_id not in USER_DB:
        USER_DB[user_id] = {}
    
    try:
        USER_DB[user_id].update(profile_update)
        # print(f"DEBUG: Updated DB for {user_id}: {USER_DB[user_id]}")
        return True, None
    except Exception as e:
        return False, str(e)

# 3. Instantiate and Use the `Signup` Handler:
async def main():
    # --- IMPORTANT ---
    # Replace "sk-your_openai_api_key" with your actual OpenAI API key
    # or ensure the OPENAI_API_KEY environment variable is set.
    # For example, to load from .env file:
    # from dotenv import load_dotenv
    # load_dotenv()
    # openai_key = os.getenv("OPENAI_API_KEY")
    # if not openai_key:
    #     print("Error: OPENAI_API_KEY not found. Please set it in your environment or .env file.")
    #     return
    # config = SignupConfig(openai_api_key=openai_key, ...)

    openai_key_to_use = os.getenv("OPENAI_API_KEY", "sk-your_openai_api_key_placeholder")
    if openai_key_to_use == "sk-your_openai_api_key_placeholder":
        print("WARNING: Using a placeholder OpenAI API key. LLM calls will likely fail.")
        print("Please set your OPENAI_API_KEY environment variable or replace the placeholder in the script.")

    config = SignupConfig(
        openai_api_key=openai_key_to_use,
        openai_default_model="gpt-4o-mini",
        # You can override default prompts here if needed
        # welcome_message="Hello there! Let's get you signed up."
    )

    signup_handler = Signup(
        user_model_cls=MyUserProfile,
        get_user_profile_func=get_my_user_profile,
        update_user_profile_func=update_my_user_profile,
        signup_config=config
    )

    user_id = "test_user_123"
    USER_DB.clear() # Clear DB for a fresh run

    print(f"Signup complete for {user_id} at start? {signup_handler.is_signup_complete(user_id)}")

    messages_to_send = [
        "Hi, I want to sign up.",
        "My name is John Doe.",
        "My email is john.doe@example.com",
        "I was born in 1990."
    ]

    for i, message in enumerate(messages_to_send):
        print(f"\nUser: {message}")
        bot_response = await signup_handler.handle_message(user_id, message)
        print(f"Bot: {bot_response}")
        
        # Check signup completion status after each message
        is_complete = signup_handler.is_signup_complete(user_id)
        print(f"Signup complete for {user_id} after message {i+1}? {is_complete}")
        
        # Optional: print current profile from DB
        current_profile, _ = get_my_user_profile(user_id)
        print(f"Current DB profile for {user_id}: {current_profile}")

        if is_complete and i < len(messages_to_send) -1 : # If signup completes early
            print("Signup completed before all example messages were sent.")
            # break # You might want to break or continue with other interactions

    print(f"\n--- Final Check ---")
    print(f"Signup complete for {user_id} at end? {signup_handler.is_signup_complete(user_id)}")
    
    final_profile_data, _ = get_my_user_profile(user_id)
    print(f"Final stored profile for {user_id}: {final_profile_data}")

    # Example of checking messages stored in memory for the session
    session_data = signup_handler.sessions.get_session(user_id)
    if session_data:
        print("\n--- Session Messages ---")
        for msg in session_data.get("messages", []):
            print(f"{msg['role']}: {msg['content']}...") # Print truncated content


if __name__ == "__main__":
    # To run this script:
    # 1. Make sure you are in the root directory of your project (signup2/).
    # 2. Install the package in editable mode: `pip install -e .`
    #    This makes the `signup_chatbot` package importable.
    # 3. Set your OPENAI_API_KEY environment variable.
    # 4. Run the script: `python examples/test.py`
    
    # If you don't want to install with `pip install -e .`, you can add the project root to sys.path
    # This is done at the top of the file now.

    asyncio.run(main())