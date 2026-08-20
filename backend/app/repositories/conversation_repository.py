import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.conversation import Citation, Conversation, Message
from app.models.document import DocumentChunk, DocumentVersion
from app.models.user import User

# Conversations are private to the user who started them — deliberately no admin
# bypass here (unlike documents): chat history is personal, and nothing today
# (no admin dashboard yet) needs an admin to read someone else's conversations.


async def list_conversations(session: AsyncSession, user: User) -> list[Conversation]:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    )
    return list(result.scalars().all())


async def get_conversation_with_messages(
    session: AsyncSession, conversation_id: uuid.UUID, user: User
) -> Conversation | None:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .options(
            selectinload(Conversation.messages)
            .selectinload(Message.citations)
            .selectinload(Citation.chunk)
            .selectinload(DocumentChunk.document_version)
            .selectinload(DocumentVersion.document)
        )
    )
    return result.scalar_one_or_none()
