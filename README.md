# LocalAI Client

A Python client library and web UI for LocalAI — an OpenAI-compatible local model server.

## Features

- **Python Client**: Full API client with chat, streaming, completions, embeddings
- **Web UI**: Dark-themed chat interface with real-time streaming
- **OpenAI Compatible**: Works with any OpenAI-compatible API endpoint

## Quick Start

### Python Client

```python
from localai_client import LocalAIClient, ChatMessage, MessageRole

client = LocalAIClient(
    base_url="http://localhost:20128/v1",
    api_key="your-api-key"
)

response = client.chat(
    model="your-model",
    messages=[ChatMessage(role=MessageRole.USER, content="Hello!")]
)
print(response.choices[0].message.content)
```

### Web UI

```bash
pip install flask httpx pydantic
python localai_client/app.py
```

Open `http://localhost:5000` in your browser.

## Configuration

Edit `localai_client/app.py` to set your endpoint:

```python
LOCALAI_BASE_URL = "http://localhost:20128/v1"
LOCALAI_API_KEY = "your-api-key"
```

## API Methods

| Method | Description |
|--------|-------------|
| `chat()` | Chat completion |
| `chat_stream()` | Streaming chat |
| `complete()` | Text completion |
| `complete_stream()` | Streaming completion |
| `embed()` | Text embeddings |
| `list_models()` | List available models |
| `health()` | Health check |

## License

MIT
