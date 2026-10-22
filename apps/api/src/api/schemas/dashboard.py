from pydantic import BaseModel, Field, field_validator


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9_.@-]+$')
    password: str = Field(min_length=1, max_length=256)

    @field_validator('username')
    @classmethod
    def normalize(cls, value):
        return value.lower()


class UserCreate(Credentials):
    password: str = Field(min_length=12, max_length=256)
    groups: list[str] = Field(default_factory=list, max_length=50)


class UserUpdate(BaseModel):
    active: bool | None = None
    groups: list[str] | None = Field(default=None, max_length=50)
    password: str | None = Field(default=None, min_length=12, max_length=256)
