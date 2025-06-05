# tests/test_memory.py
from signup_chatbot.memory import Memory


def test_save_and_get_messages():
    mem = Memory()
    mem.save_message("user1", "hello", role="user")
    mem.save_message("user1", "hi there", role="assistant")
    messages = mem.get_messages("user1")
    assert len(messages) == 2
    assert messages[0]["content"] == "hello"
    assert messages[0]["role"] == "user"
    assert "timestamp" in messages[0]


def test_session_metadata_operations():
    mem = Memory()
    mem.update_session_metadata("u2", "flag", True)
    assert mem.get_session_metadata_value("u2", "flag") is True
    mem.update_session_metadata("u2", "flag", False)
    assert mem.get_session_metadata_value("u2", "flag") is False


def test_delete_session():
    mem = Memory()
    mem.save_message("u3", "a")
    assert mem.delete_session("u3") is True
    assert mem.get_session("u3") is None