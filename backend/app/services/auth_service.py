from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import ROLE_ADMIN, ROLE_USER, User
from app.repositories import user_repository


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


async def register_user(session: AsyncSession, *, email: str, password: str) -> User:
    """Registers a new user. The very first user in the system becomes admin,
    everyone after that is a regular user — a simple bootstrap that needs no
    seed script or hardcoded credentials, at the cost of being a real footgun
    if this app were ever exposed publicly before anyone has registered. Fine
    for a demo/portfolio deployment; a real one would seed an admin instead.
    """
    if await user_repository.get_user_by_email(session, email) is not None:
        raise EmailAlreadyRegisteredError(f"{email} is already registered")

    is_first_user = await user_repository.count_users(session) == 0
    role_name = ROLE_ADMIN if is_first_user else ROLE_USER
    role = await user_repository.get_role_by_name(session, role_name)
    assert role is not None, f"role {role_name!r} was not seeded — check the auth migration"

    user = User(email=email, hashed_password=hash_password(password), role_id=role.id)
    session.add(user)
    await session.commit()
    await session.refresh(user, attribute_names=["role"])
    return user


async def authenticate_user(session: AsyncSession, *, email: str, password: str) -> User:
    user = await user_repository.get_user_by_email(session, email)
    if user is None or not user.is_active or not verify_password(password, user.hashed_password):
        raise InvalidCredentialsError("Incorrect email or password")
    return user


def issue_token_for(user: User) -> str:
    return create_access_token(user_id=user.id)
