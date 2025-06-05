# tests/test_email_validator.py
import pytest
from signup_chatbot.utils.email_validator import validate_email, EmailNotValidError


def test_validate_email_success():
    result = validate_email("USER@Example.COM")
    assert result.normalized == "USER@example.com"
    assert result.local_part == "USER"


def test_validate_email_failure():
    with pytest.raises(EmailNotValidError):
        validate_email("invalid-email")
