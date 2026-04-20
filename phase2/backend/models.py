"""
Pydantic models for the Phase 2 FastAPI microservice.
All state (conversation history + user profile) is owned by the client.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class HMO(str, Enum):
    maccabi  = "מכבי"
    meuhedet = "מאוחדת"
    clalit   = "כללית"


class Tier(str, Enum):
    gold   = "זהב"
    silver = "כסף"
    bronze = "ארד"


class Phase(str, Enum):
    collection = "collection"
    qa         = "qa"


# ── User profile ─────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    firstName: str = ""
    lastName:  str = ""
    idNumber:  str = ""          # validated 9-digit string
    gender:    str = ""
    age:       Optional[int] = None  # 0-120
    hmo:       str = ""           # one of HMO values
    hmoCardNumber: str = ""       # 9-digit
    tier:      str = ""           # one of Tier values

    @field_validator("idNumber", "hmoCardNumber", mode="before")
    @classmethod
    def must_be_digits(cls, v: Any) -> str:
        if v is None:
            return ""
        v = str(v).strip()
        return v

    def is_complete(self) -> bool:
        """True when all required fields are non-empty."""
        return all([
            self.firstName,
            self.lastName,
            self.idNumber,
            self.gender,
            self.age is not None,
            self.hmo,
            self.hmoCardNumber,
            self.tier,
        ])


# ── Message ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


# ── API request / response ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        description="Full conversation history (client-managed)"
    )
    user_profile: UserProfile = Field(
        default_factory=UserProfile,
        description="Current user profile (may be partial during collection phase)",
    )
    phase: Phase = Phase.collection
    language: str = "he"   # "he" | "en"


class ChatResponse(BaseModel):
    reply: str
    user_profile: UserProfile
    phase: Phase
    error: Optional[str] = None


# ── Knowledge chunk (internal, not exposed via API) ───────────────────────────

class KnowledgeChunk(BaseModel):
    category: str       # e.g. "רפואה משלימה"
    service:  str       # e.g. "דיקור סיני"
    hmo:      str       # e.g. "מכבי"
    tier:     str       # e.g. "זהב"
    content:  str       # full text for embedding + retrieval
    embedding: List[float] = Field(default_factory=list)
