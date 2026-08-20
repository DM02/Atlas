from __future__ import annotations

import uuid

from pydantic import BaseModel, EmailStr, Field

from app.models.user import User


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: str

    @classmethod
    def from_user(cls, user: User) -> UserOut:
        return cls(id=user.id, email=user.email, role=user.role.name)
