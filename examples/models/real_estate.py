# examples/models/real_estate.py
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr, PositiveInt, ConfigDict, field_validator
from enum import Enum

# --- Enums ---
class TransactionType(str, Enum):
    RENT = "rent"
    SALE = "buy"

class PropertyType(str, Enum):
    APARTMENT = "apartment"
    HOUSE = "house"
    LAND = "land"
    COMMERCIAL = "commercial"

# --- Determine static year bounds at module load time ---
CURRENT_YEAR = datetime.now().year
MAX_AGE = 120
MIN_AGE = 16
OLDEST_POSSIBLE_BIRTH_YEAR = CURRENT_YEAR - MAX_AGE
YOUNGEST_POSSIBLE_BIRTH_YEAR = CURRENT_YEAR - MIN_AGE


# --- Signup and UserProfile Models ---
class SignupCoreInfo(BaseModel):
    user_id: str = Field(..., description="Unique identifier (e.g., 'whatsapp:+11234567890').")
    name: Optional[str] = Field(None, description="User's full name.")
    email: Optional[EmailStr] = Field(None, description="User's primary email address.")
    birth_year: Optional[PositiveInt] = Field(
        None,
        description="User's birth year (e.g., 1990).",
        ge=OLDEST_POSSIBLE_BIRTH_YEAR,
        le=YOUNGEST_POSSIBLE_BIRTH_YEAR
    )
    phone_number: Optional[str] = Field(
        None,
        description="User's contact phone number.",
        min_length=7,
        pattern=r"^\+?[0-9\s\-\(\)]{7,20}$"
    )
    neighborhood: Optional[str] = Field(None, description="User's neighborhood of residence.")
    city: Optional[str] = Field(None, description="User's primary city of residence.")

    model_config = ConfigDict(
        use_enum_values=True,
        extra='ignore',
        validate_assignment=True
    )

class UserProfile(SignupCoreInfo):
    primary_transaction_type: Optional[TransactionType] = Field(None, description="Typical goal (buy or rent).")
    preferred_property_types: Optional[List[PropertyType]] = Field(
        None,
        description="User's preferred property types."
    )
    general_preferences: Optional[List[str]] = Field(
        None,
        description="Other recurring needs (e.g., 'pet-friendly', 'garage')."
    )
    signup_completed: bool = Field(
        default=False, 
        description="Flag indicating if the signup process has been completed."
    )
    first_interaction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_interaction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        use_enum_values=True,
        extra='ignore',
        validate_assignment=True
    )

    @classmethod
    def default(cls, user_id: str):
        now_utc = datetime.now(timezone.utc)
        return cls(
            user_id=user_id,
            first_interaction_timestamp=now_utc,
            last_interaction_timestamp=now_utc,
            optional_info_collection_completed=False # Explicitly set default
        )

    def update_last_interaction(self):
        self.last_interaction_timestamp = datetime.now(timezone.utc)

    @field_validator('birth_year')
    @classmethod
    def check_birth_year_range(cls, v: Optional[int]) -> Optional[int]:
        if v is None: return v
        current_year = datetime.now().year
        oldest_birth_year = current_year - MAX_AGE
        youngest_birth_year = current_year - MIN_AGE
        if not (oldest_birth_year <= v <= youngest_birth_year):
            raise ValueError(f"Birth year must be between {oldest_birth_year} and {youngest_birth_year}")
        return v