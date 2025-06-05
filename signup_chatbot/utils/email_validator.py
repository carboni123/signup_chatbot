# signup_chatbot/utils/email_validator.py
class EmailNotValidError(ValueError):
    """Exception raised when an email address is not valid."""

class ValidatedEmail:
    def __init__(self, normalized: str, local_part: str):
        self.normalized = normalized
        self.local_part = local_part

# Very naive email check just to satisfy pydantic during tests
def validate_email(email: str, check_deliverability: bool = False):
    if "@" not in email:
        raise EmailNotValidError("Invalid email address")
    local_part, domain = email.split("@", 1)
    normalized = f"{local_part}@{domain.lower()}"
    return ValidatedEmail(normalized=normalized, local_part=local_part)
