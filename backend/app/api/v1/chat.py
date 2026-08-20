import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.embeddings.factory import get_embedding_provider
from app.ai.errors import AIProviderError
from app.ai.llm.base import LLMProvider
from app.ai.llm.factory import get_llm_provider
from app.ai.reranker.base import RerankerProvider
from app.ai.reranker.factory import get_reranker
from app.api.deps import get_current_user
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.user import User
from app.repositories import conversation_repository
from app.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    CitationOut,
    ConversationDetailOut,
    ConversationOut,
    MessageOut,
)
from app.services.chat_service import ConversationNotFoundError, answer_query

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatQueryResponse)
async def query(
    payload: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    llm: LLMProvider = Depends(get_llm_provider),
    reranker: RerankerProvider = Depends(get_reranker),
) -> ChatQueryResponse:
    try:
        result = await answer_query(
            session=session,
            embedding_provider=embedding_provider,
            llm=llm,
            settings=settings,
            query=payload.query,
            conversation_id=payload.conversation_id,
            requesting_user=current_user,
            reranker=reranker,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI provider is currently unavailable: {exc}",
        ) from exc

    return ChatQueryResponse(
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                version_number=c.version_number,
                page_number=c.page_number,
                section_title=c.section_title,
                score=c.score,
            )
            for c in result.citations
        ],
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_db)
) -> list[ConversationOut]:
    conversations = await conversation_repository.list_conversations(session, current_user)
    return [ConversationOut.model_validate(c) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationDetailOut:
    conversation = await conversation_repository.get_conversation_with_messages(
        session, conversation_id, current_user
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
            citations=[
                CitationOut(
                    chunk_id=c.chunk_id,
                    document_id=c.chunk.document_version.document.id,
                    document_title=c.chunk.document_version.document.title,
                    version_number=c.chunk.document_version.version_number,
                    page_number=c.page,
                    section_title=c.section,
                    score=c.score or 0.0,
                )
                for c in m.citations
            ],
        )
        for m in conversation.messages
    ]

    return ConversationDetailOut(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=messages,
    )
