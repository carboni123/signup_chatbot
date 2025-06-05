# tests/test_signup.py
import pytest
from pydantic import BaseModel
from signup_chatbot.signup import Signup
from signup_chatbot.config_models import SignupConfig


class SimpleProfile(BaseModel):
    user_id: str
    name: str | None = None
    age: int | None = None


USER_DB: dict[str, dict] = {}


def get_profile(user_id: str):
    if user_id in USER_DB:
        return USER_DB[user_id].copy(), None
    return None, None


def update_profile(user_id: str, update: dict):
    USER_DB.setdefault(user_id, {})
    USER_DB[user_id].update(update)
    return True, None


def test_dynamic_model_fields():
    signup = Signup(
        user_model_cls=SimpleProfile,
        get_user_profile_func=get_profile,
        update_user_profile_func=update_profile,
        signup_config=SignupConfig(),
        user_model_fields_to_ignore=["user_id"],
    )
    dynamic_model = signup._llm_visible_profile_model_cls
    assert "user_id" not in dynamic_model.model_fields
    assert set(dynamic_model.model_fields.keys()) == {"name", "age"}


def test_is_signup_complete_and_update(dummy_llm_client):
    USER_DB.clear()
    signup = Signup(
        user_model_cls=SimpleProfile,
        get_user_profile_func=get_profile,
        update_user_profile_func=update_profile,
        signup_config=SignupConfig(),
        user_model_fields_to_ignore=["user_id"],
    )
    assert not signup.is_signup_complete("u1")
    # simulate filling profile
    update_profile("u1", {"name": "Alice", "age": 30})
    assert signup.is_signup_complete("u1")


def test_handle_message_basic(dummy_llm_client):
    USER_DB.clear()
    dummy_llm_client.response = "Response from LLM"
    signup = Signup(
        user_model_cls=SimpleProfile,
        get_user_profile_func=get_profile,
        update_user_profile_func=update_profile,
        signup_config=SignupConfig(),
        user_model_fields_to_ignore=["user_id"],
    )
    import asyncio

    replies = asyncio.run(signup.handle_message("u2", "hello"))
    assert dummy_llm_client.calls
    assert any("Response from LLM" in r for r in replies)
    # history should contain user and assistant message
    hist = signup.sessions.get_messages("u2")
    assert hist and hist[-1]["role"] == "assistant"


def test_skip_command(dummy_llm_client):
    USER_DB.clear()
    signup = Signup(
        user_model_cls=SimpleProfile,
        get_user_profile_func=get_profile,
        update_user_profile_func=update_profile,
        signup_config=SignupConfig(skip_command="/skip"),
        user_model_fields_to_ignore=["user_id"],
    )

    import asyncio

    replies = asyncio.run(signup.handle_message("u3", "/skip"))
    assert not dummy_llm_client.calls  # LLM should not be invoked
    assert signup.is_signup_complete("u3")
    assert any(signup.config.signup_complete_message in r for r in replies)
