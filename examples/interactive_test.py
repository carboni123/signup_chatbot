# examples/interactive_test.py
import asyncio
import os
import logging
import pytest
from pydantic import ValidationError

pytest.skip("Examples are not part of the test suite", allow_module_level=True)
from typing import Optional, Tuple, Dict, Any

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signup_chatbot import Signup, SignupConfig
from examples.models.real_estate import UserProfile

# Configure basic logging for the example
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# You might want to set the signup_chatbot logger to DEBUG for more detailed output
# logging.getLogger("signup_chatbot").setLevel(logging.DEBUG)


# --- Mock Database and Profile Functions ---
USER_DB: Dict[str, Dict[str, Any]] = {} # Stores serialized UserProfile data

def get_my_user_profile(user_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Retrieves the user's profile from the mock DB.
    Returns the profile as a dictionary.
    """
    # logging.debug(f"PROFILE_DB_GET: Called for {user_id}. Current DB: {USER_DB.get(user_id)}")
    if user_id in USER_DB:
        return USER_DB[user_id].copy(), None
    return None, None # No profile exists yet

def update_my_user_profile(user_id: str, profile_update: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Updates the user's profile in the mock DB.
    'profile_update' is a dictionary of fields to update.
    """
    logging.info(f"PROFILE_DB_UPDATE: Updating profile for {user_id} with: {profile_update}")
    if user_id not in USER_DB:
        # If user doesn't exist, initialize with a default UserProfile structure based on their ID
        # then apply the update. This ensures 'first_interaction_timestamp' etc. are set.
        try:
            # Create a default instance, then get its dict representation
            default_profile = UserProfile.default(user_id=user_id)
            USER_DB[user_id] = default_profile.model_dump(exclude_none=True, mode="json") # Store as dict
        except Exception as e:
            logging.error(f"PROFILE_DB_UPDATE: Error creating default profile for {user_id}: {e}")
            return False, f"Error initializing profile: {e}"

    try:
        # Apply updates to the existing or newly created profile dictionary
        # We need to be careful here: profile_update might contain values that need Pydantic validation/conversion
        # For a more robust mock, we'd load the dict into UserProfile, update, then dump back.
        
        current_profile_dict = USER_DB[user_id]
        temp_profile_data = current_profile_dict.copy()
        temp_profile_data.update(profile_update)

        # Validate the merged data by trying to load it into the model
        # This ensures that updates are type-correct and pass model validation (e.g., birth_year range)
        validated_profile = UserProfile(**temp_profile_data)
        USER_DB[user_id] = validated_profile.model_dump(exclude_none=True, mode="json") # Store the validated and dumped dict

        logging.info(f"PROFILE_DB_UPDATE: Successfully updated profile for {user_id}. New state: {USER_DB[user_id]}")
        return True, None
    except ValidationError as e:
        logging.error(f"PROFILE_DB_UPDATE: Validation error updating profile for {user_id}: {e}. Update: {profile_update}")
        return False, f"Invalid data for update: {e.errors()}"
    except Exception as e:
        logging.error(f"PROFILE_DB_UPDATE: Unexpected error updating profile for {user_id}: {e}")
        return False, str(e)

async def interactive_main_loop():
    # --- Configuration ---
    openai_key_to_use = os.getenv("OPENAI_API_KEY")
    if not openai_key_to_use:
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("LLM calls will fail. Please set it and restart.")
        # return # You might want to exit if key is crucial

    config = SignupConfig(
        openai_api_key=openai_key_to_use,
        openai_default_model="gpt-4o-mini", # Or your preferred model
        # llm_system_prompt_template: (Use default or customize here)
    )

    # --- Initialization ---
    # Ensure UserProfile model is used (update signup.py if necessary for field checking)
    signup_handler = Signup(
        user_model_cls=UserProfile, # Use the detailed UserProfile model
        get_user_profile_func=get_my_user_profile,
        update_user_profile_func=update_my_user_profile,
        signup_config=config,
        user_model_fields_to_ignore=["user_id"]
    )

    user_id = input("Enter a User ID (e.g., test_user_interactive): ").strip()
    if not user_id:
        user_id = "test_user_interactive"
    print(f"Using User ID: {user_id}")
    USER_DB.clear() # Clear DB for a fresh run with this user_id

    print("\nSignup Chatbot Interactive Test")
    print("Type 'quit' or 'exit' to end.")
    print("Type '/profile' to see your current stored profile.")
    print("Type '/checkcomplete' to see if signup is marked complete.")
    print("--------------------------------------------------")

    # Initial message to kick off the conversation if profile is empty
    # or let the LLM guide based on the system prompt and current profile state.
    # Forcing an initial message can sometimes be useful.
    # initial_bot_message = await signup_handler.handle_message(user_id, "Hello") # Optional: Seed with a greeting
    # print(f"Bot: {initial_bot_message}")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Goodbye!")
            break
        
        if user_input.lower() == '/profile':
            profile_data, err = get_my_user_profile(user_id)
            if err:
                print(f"Bot: Error fetching profile: {err}")
            elif profile_data:
                print("\n--- Current Stored Profile ---")
                # Pretty print the dictionary
                import json
                print(json.dumps(profile_data, indent=2))
                print("-----------------------------")
            else:
                print("Bot: No profile data found.")
            continue

        if user_input.lower() == '/checkcomplete':
            is_complete = signup_handler.is_signup_complete(user_id)
            db_profile_dict, _ = get_my_user_profile(user_id)
            db_signup_completed_flag = db_profile_dict.get("signup_completed", False) if db_profile_dict else False
            
            print(f"Bot: Signup complete (according to missing fields): {is_complete}")
            print(f"Bot: Signup_completed flag in DB: {db_signup_completed_flag}") # Check the actual flag
            continue
            
        if not user_input:
            continue

        # --- Process Message ---
        print("Bot: Thinking...")
        bot_response = await signup_handler.handle_message(user_id, user_input,interaction_context="returning_user")
        for message in bot_response:
            print(f"Bot: {message}")
            await asyncio.sleep(1)

        # Check if signup is complete after each interaction
        if signup_handler.is_signup_complete(user_id):
            print("Bot: (Info: All required profile fields seem to be filled according to the model!)")
            return


if __name__ == "__main__":
    # Ensure you are in the root directory (signup2) and have run `pip install -e .`
    # Also, set your OPENAI_API_KEY environment variable.
    try:
        asyncio.run(interactive_main_loop())
    except KeyboardInterrupt:
        print("\nExiting...")