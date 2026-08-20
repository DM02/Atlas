import uuid

from arq import ArqRedis
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_admin
from app.core.config import Settings, get_settings
from app.core.queue import get_arq_pool
from app.core.storage import StorageBackend, get_storage_backend
from app.db.session import get_db
from app.models.user import User
from app.repositories import document_repository
from app.schemas.document import ChunkOut, DocumentOut, DocumentStatusOut, DocumentVersionOut
from app.services.ingestion_service import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    create_pending_version,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    private: bool = Form(False),
    current_user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    settings: Settings = Depends(get_settings),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> DocumentOut:
    content = await file.read()

    try:
        version = await create_pending_version(
            session=session,
            storage=storage,
            settings=settings,
            filename=file.filename or "untitled",
            content=content,
            owner_id=current_user.id,
            is_private=private,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc

    await arq_pool.enqueue_job("run_ingestion", str(version.id))

    document = await document_repository.get_document(session, version.document_id, current_user)
    assert document is not None
    return DocumentOut.model_validate(document)


@router.put(
    "/{document_id}", response_model=DocumentVersionOut, status_code=status.HTTP_202_ACCEPTED
)
async def upload_new_version(
    document_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    settings: Settings = Depends(get_settings),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> DocumentVersionOut:
    """Re-upload: adds a new version to an existing document and queues it for
    processing. The document keeps serving whatever version is already current
    until this one finishes successfully — see ingestion_service.process_document_version.
    """
    document = await document_repository.get_document(session, document_id, current_user)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    content = await file.read()

    try:
        version = await create_pending_version(
            session=session,
            storage=storage,
            settings=settings,
            filename=file.filename or document.title,
            content=content,
            owner_id=current_user.id,
            document=document,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc

    await arq_pool.enqueue_job("run_ingestion", str(version.id))

    # Not hardcoded: with a real (async) queue this job hasn't run yet, so it's
    # false. But arq_pool is swappable (tests use one that runs the job inline
    # before enqueue_job even returns), so re-check reality instead of assuming.
    await session.refresh(document, attribute_names=["current_version_id"])
    return DocumentVersionOut(
        id=version.id,
        version_number=version.version_number,
        created_at=version.created_at,
        is_current=document.current_version_id == version.id,
    )


@router.get("/{document_id}/versions", response_model=list[DocumentVersionOut])
async def list_document_versions(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[DocumentVersionOut]:
    document = await document_repository.get_document(session, document_id, current_user)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    versions = await document_repository.list_versions(session, document_id)
    return [
        DocumentVersionOut(
            id=v.id,
            version_number=v.version_number,
            created_at=v.created_at,
            is_current=(v.id == document.current_version_id),
        )
        for v in versions
    ]


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_db)
) -> list[DocumentOut]:
    documents = await document_repository.list_documents(session, current_user)
    return [DocumentOut.model_validate(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> DocumentOut:
    document = await document_repository.get_document(session, document_id, current_user)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentOut.model_validate(document)


@router.get("/{document_id}/status", response_model=DocumentStatusOut)
async def get_document_status(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> DocumentStatusOut:
    document = await document_repository.get_document(session, document_id, current_user)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    latest = await document_repository.get_latest_ingestion_job(session, document_id)
    job, version_number = latest if latest else (None, None)
    return DocumentStatusOut(
        id=document.id,
        status=document.status,
        latest_version_number=version_number,
        error_message=job.error_message if job else None,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> None:
    document = await document_repository.get_document(session, document_id, current_user)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await document_repository.delete_document(session, document)


@router.get("/{document_id}/chunks/{chunk_id}", response_model=ChunkOut)
async def get_chunk(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ChunkOut:
    row = await document_repository.get_chunk_with_document(
        session, document_id, chunk_id, current_user
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")

    chunk, version, document = row
    return ChunkOut(
        id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        version_number=version.version_number,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
    )
