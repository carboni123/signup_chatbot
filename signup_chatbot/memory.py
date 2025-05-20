from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

class Memory:
    """
    In-memory storage for user session information.
    Each session can store a list of messages. Message entries can include their own metadata.
    """
    def __init__(self) -> None:
        """
        Initializes the memory store.
        The internal structure is a dictionary mapping user_id to their session data.
        Session data is a dictionary, primarily including a 'messages' list.
        A session-level 'metadata' dictionary is also maintained for general session state if needed,
        but message-specific metadata is now stored within each message entry.
        
        Example structure of self.memory:
        {
            "user123": {
                "messages": [
                    {
                        "role": "user", 
                        "content": "Hello", 
                        "timestamp": "2023-10-27T10:00:00Z",
                        "metadata": {"source": "cli"} # Example message-specific metadata
                    },
                    {
                        "role": "assistant", 
                        "content": "Hi there! How can I help?", 
                        "timestamp": "2023-10-27T10:00:05Z",
                        "metadata": {"tool_calls": [...]} # Example message-specific metadata
                    }
                ],
                "session_metadata": {"some_general_session_flag": True} # Renamed for clarity
            },
            "user456": { ... }
        }
        """
        self.memory: Dict[str, Dict[str, Any]] = {}

    def _ensure_session_structure(self, user_id: str) -> Dict[str, Any]:
        """
        Ensures a session dictionary exists for the user_id and has the basic structure
        ('messages' list and 'session_metadata' dict). Creates it if necessary.
        Returns the session dictionary for the given user_id.
        """
        if user_id not in self.memory:
            self.memory[user_id] = {
                "messages": [],
                "session_metadata": {} # For general session state, distinct from message metadata
            }
        else:
            session = self.memory[user_id]
            if "messages" not in session or not isinstance(session.get("messages"), list):
                session["messages"] = []
            if "session_metadata" not in session or not isinstance(session.get("session_metadata"), dict):
                session["session_metadata"] = {}
        return self.memory[user_id]

    def __call__(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get method for user session.
        Returns the entire session dictionary for the user, or None if the user_id is not found.
        """
        return self.memory.get(user_id)

    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Explicitly named method to get a user's entire session data. Alias for __call__.
        """
        return self.__call__(user_id)

    def save_message(
        self,
        user_id: str,
        content: str,
        role: str = "user",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None  # Added metadata argument
    ) -> None:
        """
        Saves a structured message to the user's session history.
        If the session doesn't exist, it's created.
        Message-specific metadata can be included.

        Args:
            user_id: The unique identifier for the user.
            content: The textual content of the message.
            role: The role of the entity sending the message.
            timestamp: Optional ISO format string timestamp. If None, current UTC time is used.
            metadata: Optional dictionary for message-specific metadata.
        """
        session = self._ensure_session_structure(user_id)
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        message_entry: Dict[str, Any] = { # Ensure type checker knows about metadata
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        if metadata is not None:
            message_entry["metadata"] = metadata
        
        session["messages"].append(message_entry)

    def get_messages(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the list of messages for a given user.
        Returns an empty list if the user session or messages list doesn't exist.
        """
        session = self.get_session(user_id)
        if session:
            return session.get("messages", []) 
        return []

    # --- Methods for session-level metadata (if general session state is still needed) ---
    def update_session_metadata(self, user_id: str, key: str, value: Any) -> None:
        """
        Updates or adds a key-value pair to the user's general 'session_metadata' dictionary.
        This is for metadata not specific to a single message.
        """
        session = self._ensure_session_structure(user_id)
        session["session_metadata"][key] = value
        
    def get_session_metadata_value(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Retrieves a specific value from the user's 'session_metadata'.

        Args:
            user_id: The unique identifier for the user.
            key: The key of the session metadata to retrieve.
            default: The value to return if not found.
        Returns:
            The session metadata value, or the specified default.
        """
        session = self.get_session(user_id)
        if session and "session_metadata" in session:
            return session["session_metadata"].get(key, default)
        return default
    # --- End of session-level metadata methods ---

    def delete_session(self, user_id: str) -> bool:
        """
        Deletes the entire session for a given user.
        """
        if user_id in self.memory:
            del self.memory[user_id]
            return True
        return False

    def clear_user_messages(self, user_id: str) -> bool:
        """
        Clears all messages for a specific user but keeps their session active 
        (i.e., session_metadata is preserved).
        """
        session = self.get_session(user_id)
        if session:
            session["messages"] = []
            return True
        return False

    def clear_all_sessions(self) -> None:
        """
        Clears all stored sessions from memory.
        """
        self.memory.clear()

    def list_active_session_ids(self) -> List[str]:
        """
        Returns a list of all user_ids that currently have an active session in memory.
        """
        return list(self.memory.keys())

# --- Example Usage ---
if __name__ == "__main__":
    chat_memory = Memory()

    user_alice = "alice_001"
    user_bob = "bob_002"

    # Alice starts a session
    chat_memory.save_message(user_alice, "Hi, I'd like to sign up.", role="user", metadata={"source": "web_chat"})
    # Example of setting session-level metadata if still needed
    chat_memory.update_session_metadata(user_alice, "signup_flow_active", True)
    chat_memory.update_session_metadata(user_alice, "current_step", "greeting")


    # Assistant responds to Alice, with message-specific metadata
    assistant_response_metadata = {"llm_model": "gpt-4o-mini", "processing_time_ms": 120}
    chat_memory.save_message(
        user_alice, 
        "Hello Alice! Great. What's your email?", 
        role="assistant", 
        metadata=assistant_response_metadata
    )
    chat_memory.update_session_metadata(user_alice, "current_step", "collecting_email")
    
    # Bob starts a session
    chat_memory.save_message(user_bob, "I need help with my account.", role="user")
    chat_memory.update_session_metadata(user_bob, "topic", "account_support")


    # Get Alice's session data
    alice_session = chat_memory.get_session(user_alice)
    if alice_session:
        print(f"\nAlice's Session ({user_alice}):")
        print("  Messages:")
        for msg in alice_session.get("messages", []):
            print(f"    - Role: {msg['role']}, Content: '{msg['content']}', Meta: {msg.get('metadata')}")
        print(f"  Session Metadata: {alice_session.get('session_metadata')}")
        print(f"  Current signup step (session metadata): {chat_memory.get_session_metadata_value(user_alice, 'current_step')}")

    # Get Bob's messages
    bob_messages = chat_memory.get_messages(user_bob)
    print(f"\nBob's Messages ({user_bob}):")
    for msg in bob_messages:
        print(f"  - Role: {msg['role']}, Content: '{msg['content']}', Meta: {msg.get('metadata')}")


    # List active sessions
    print(f"\nActive session IDs: {chat_memory.list_active_session_ids()}")

    # Alice completes signup (example interaction)
    chat_memory.save_message(user_alice, "alice@example.com", role="user")
    chat_memory.update_session_metadata(user_alice, "email_collected", True)
    chat_memory.update_session_metadata(user_alice, "current_step", "completed")
    print(f"\nAlice's current step (session metadata) after email: {chat_memory.get_session_metadata_value(user_alice, 'current_step')}")

    # Delete Alice's session
    if chat_memory.delete_session(user_alice):
        print(f"\nSession for {user_alice} deleted.")
    
    alice_session_after_delete = chat_memory.get_session(user_alice)
    print(f"Alice's session after delete: {alice_session_after_delete}")

    print(f"\nActive session IDs after deleting Alice: {chat_memory.list_active_session_ids()}")

    chat_memory.clear_all_sessions()
    print(f"\nActive session IDs after clearing all: {chat_memory.list_active_session_ids()}")