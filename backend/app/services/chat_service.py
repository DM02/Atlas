import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.errors import AIProviderError
from app.ai.llm.base import LLMProvider
from app.ai.pipeline.rag_pipeline import generate_answer, parse_cited_indices, stream_answer
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


@dataclass
class PreparedChat:
    conversation_id: uuid.UUID
    chunks: list[RetrievedChunk]
    retrieve_ms: float


@dataclass
class TokenEvent:
    text: str


@dataclass
class DoneEvent:
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    citations: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class ErrorEvent:
    detail: str


StreamEvent = TokenEvent | DoneEvent | ErrorEvent


async def _resolve_conversation(
    session: AsyncSession, conversation_id: uuid.UUID | None, requesting_user: User, query: str
) -> Conversation:
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
    return conversation


async def prepare_chat(
    *,
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    query: str,
    conversation_id: uuid.UUID | None,
    requesting_user: User,
    reranker: RerankerProvider | None = None,
) -> PreparedChat:
    """Everything that must still raise a normal HTTP error status: resolving/
    creating the conversation and retrieving chunks. Called before the
    streaming response starts, since headers can't change once it has.
    """
    conversation = await _resolve_conversation(session, conversation_id, requesting_user, query)
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
    await session.commit()

    return PreparedChat(conversation_id=conversation.id, chunks=chunks, retrieve_ms=retrieve_ms)


async def stream_chat(
    *, session: AsyncSession, llm: LLMProvider, query: str, prepared: PreparedChat
) -> AsyncIterator[StreamEvent]:
    """Streams the LLM's answer token-by-token, then persists the assistant
    message/citations once the stream ends. A provider error mid-stream can't
    become an HTTP error anymore (the 200 status is already sent), so it's
    surfaced as an in-band ErrorEvent instead — same as ChatResult, nothing is
    persisted for a failed generation.
    """
    generate_start = time.perf_counter()
    answer_parts: list[str] = []
    try:
        async for delta in stream_answer(llm=llm, query=query, chunks=prepared.chunks):
            answer_parts.append(delta)
            yield TokenEvent(text=delta)
    except AIProviderError as exc:
        yield ErrorEvent(detail=str(exc))
        return

    answer = "".join(answer_parts)
    generate_ms = (time.perf_counter() - generate_start) * 1000
    total_ms = prepared.retrieve_ms + generate_ms

    logger.info(
        "chat_query_completed",
        retrieve_ms=round(prepared.retrieve_ms, 1),
        generate_ms=round(generate_ms, 1),
        total_ms=round(total_ms, 1),
        num_chunks=len(prepared.chunks),
    )
    session.add(
        RequestMetric(
            endpoint="chat_query",
            stage_latencies_ms={
                "retrieve_ms": round(prepared.retrieve_ms, 1),
                "generate_ms": round(generate_ms, 1),
            },
            total_ms=round(total_ms),
        )
    )

    assistant_message = Message(
        conversation_id=prepared.conversation_id, role=MESSAGE_ROLE_ASSISTANT, content=answer
    )
    session.add(assistant_message)
    await session.flush()

    cited_indices = parse_cited_indices(answer, max_index=len(prepared.chunks))
    cited_chunks = [prepared.chunks[i - 1] for i in cited_indices]
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

    yield DoneEvent(
        conversation_id=prepared.conversation_id,
        message_id=assistant_message.id,
        citations=cited_chunks,
    )


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
    conversation = await _resolve_conversation(session, conversation_id, requesting_user, query)
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
