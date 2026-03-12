import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService


router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about indexed documents.")


class SourceReference(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)


class StreamChunk(BaseModel):
    type: str
    content: str | None = None
    sources: list[SourceReference] | None = None


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        response = chat_service.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return ChatResponse(**response)


@router.post("/chat/stream", status_code=status.HTTP_200_OK)
async def chat_stream(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    try:
        stream, sources = chat_service.ask_stream(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    def event_stream():
        logger.info("Streaming chat response.", extra={"question_length": len(payload.question)})
        for chunk in stream:
            message = StreamChunk(type="chunk", content=chunk)
            yield f"data: {message.model_dump_json()}\n\n"

        final_message = StreamChunk(type="sources", sources=[SourceReference(**source) for source in sources])
        yield f"data: {final_message.model_dump_json()}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")