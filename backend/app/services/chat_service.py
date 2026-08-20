import time
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.llm.base import LLMProvider
from app.ai.pipeline.rag_pipeline import generate_answer
from app.ai.reranker.base import RerankerProvider
from app.core.config import Settings
from app.models.conversation import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    Citation,
    Conversation,
    Message,
)
from app.models.metrics import RequestMetric
from app.models.user import User
from app.services.retrieval_service import RetrievedChunk, retrieve_relevant_chunks

logger = structlog.get_logger(__name__)


class ConversationNotFoundError(Exception):
    pass


@dataclass
class ChatResult:
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[RetrievedChunk]


async def answer_query(
    *,
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    llm: LLMProvider,
    settings: Settings,
    query: str,
    conversation_id: uuid.UUID | None,
    requesting_user: User,
    reranker: RerankerProvider | None = None,
) -> ChatResult:
    if conversation_id is not None:
        conversation = await session.get(Conversation, conversation_id)
        # Same error for "doesn't exist" and "exists but isn't yours" — a 404 instead
        # of a 403 doesn't confirm to a caller that someone else's conversation exists.
        if conversation is None or conversation.user_id != requesting_user.id:
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
    else:
        conversation = Conversation(user_id=requesting_user.id, title=query[:200])
        session.add(conversation)
        await session.flush()

    session.add(Message(conversation_id=conversation.id, role=MESSAGE_ROLE_USER, content=query))

    start = time.perf_counter()
    chunks = await retrieve_relevant_chunks(
        session=session,
        embedding_provider=embedding_provider,
        query=query,
        top_k=settings.retrieval_top_k,
        use_hybrid_search=settings.enable_hybrid_search,
        use_reranking=settings.enable_reranking,
        reranker=reranker,
        candidate_pool_size=settings.retrieval_candidate_pool_size,
        requesting_user=requesting_user,
    )
    retrieve_ms = (time.perf_counter() - start) * 1000

    generate_start = time.perf_counter()
    generation = await generate_answer(llm=llm, query=query, chunks=chunks)
    generate_ms = (time.perf_counter() - generate_start) * 1000
    total_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "chat_query_completed",
        retrieve_ms=round(retrieve_ms, 1),
        generate_ms=round(generate_ms, 1),
        total_ms=round(total_ms, 1),
        num_chunks=len(chunks),
    )
    session.add(
        RequestMetric(
            endpoint="chat_query",
            stage_latencies_ms={
                "retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": round(generate_ms, 1),
            },
            total_ms=round(total_ms),
        )
    )

    assistant_message = Message(
        conversation_id=conversation.id, role=MESSAGE_ROLE_ASSISTANT, content=generation.answer
    )
    session.add(assistant_message)
    await session.flush()

    cited_chunks = [chunks[i - 1] for i in generation.cited_chunk_indices]
    for chunk in cited_chunks:
        session.add(
            Citation(
                message_id=assistant_message.id,
                chunk_id=chunk.chunk_id,
                page=chunk.page_number,
                section=chunk.section_title,
                score=chunk.score,
            )
        )

    await session.commit()

    return ChatResult(
        conversation_id=conversation.id,
        message_id=assistant_message.id,
        answer=generation.answer,
        citations=cited_chunks,
    )
