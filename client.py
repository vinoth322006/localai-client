import json
from typing import Iterator, Optional, Union

import httpx

from .exceptions import APIError, ConnectionError, ModelNotFoundError, TimeoutError
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChoiceMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    ModelListResponse,
    ModelObject,
    StreamingChunk,
    Usage,
)


class LocalAIClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    def _handle_error(self, response: httpx.Response):
        if response.status_code == 404:
            raise ModelNotFoundError(f"Not found: {response.url}")
        if response.status_code >= 400:
            raise APIError(
                message=f"API error {response.status_code}: {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )

    def _parse_sse(self, response: httpx.Response) -> Iterator[StreamingChunk]:
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data)
                    yield StreamingChunk(**chunk_data)
                except json.JSONDecodeError:
                    continue

    def health(self) -> dict:
        response = self.client.get("/health")
        self._handle_error(response)
        return response.json()

    def list_models(self) -> ModelListResponse:
        response = self.client.get("/v1/models")
        self._handle_error(response)
        data = response.json()
        return ModelListResponse(
            data=[ModelObject(**m) for m in data.get("data", [])]
        )

    def delete_model(self, model: str) -> bool:
        response = self.client.delete(f"/v1/models/{model}")
        self._handle_error(response)
        return response.status_code == 200

    def chat(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Iterator[StreamingChunk]]:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            **kwargs,
        )
        response = self.client.post(
            "/v1/chat/completions",
            json=request.model_dump(exclude_none=True),
        )
        self._handle_error(response)

        if stream:
            return self._parse_sse(response)

        data = response.json()
        return ChatCompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=[
                ChatChoice(
                    index=c.get("index", 0),
                    message=ChoiceMessage(**c.get("message", {})),
                    finish_reason=c.get("finish_reason"),
                )
                for c in data.get("choices", [])
            ],
            usage=Usage(**data.get("usage", {})),
        )

    def chat_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs,
    ) -> Iterator[StreamingChunk]:
        return self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            **kwargs,
        )

    def complete(
        self,
        model: str,
        prompt: Union[str, list],
        temperature: float = 0.7,
        max_tokens: int = 16,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> Union[CompletionResponse, Iterator[StreamingChunk]]:
        request = CompletionRequest(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            **kwargs,
        )
        response = self.client.post(
            "/v1/completions",
            json=request.model_dump(exclude_none=True),
        )
        self._handle_error(response)

        if stream:
            return self._parse_sse(response)

        data = response.json()
        return CompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=[
                CompletionChoice(
                    index=c.get("index", 0),
                    text=c.get("text", ""),
                    finish_reason=c.get("finish_reason"),
                )
                for c in data.get("choices", [])
            ],
            usage=Usage(**data.get("usage", {})),
        )

    def complete_stream(
        self,
        model: str,
        prompt: Union[str, list],
        temperature: float = 0.7,
        max_tokens: int = 16,
        top_p: float = 1.0,
        **kwargs,
    ) -> Iterator[StreamingChunk]:
        return self.complete(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            **kwargs,
        )

    def embed(
        self,
        model: str,
        input: Union[str, list],
        **kwargs,
    ) -> EmbeddingResponse:
        request = EmbeddingRequest(model=model, input=input, **kwargs)
        response = self.client.post(
            "/v1/embeddings",
            json=request.model_dump(exclude_none=True),
        )
        self._handle_error(response)
        data = response.json()
        return EmbeddingResponse(
            object=data.get("object", "list"),
            data=[EmbeddingData(**d) for d in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(**data.get("usage", {})),
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
