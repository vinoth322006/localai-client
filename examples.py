from localai_client import LocalAIClient, ChatMessage, MessageRole


def main():
    client = LocalAIClient(base_url="http://localhost:8080")

    print("=== Health Check ===")
    try:
        health = client.health()
        print(f"Server status: {health}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    print("\n=== List Models ===")
    try:
        models = client.list_models()
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as e:
        print(f"Failed to list models: {e}")

    print("\n=== Chat Completion ===")
    try:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
        ]
        response = client.chat(model="gpt-3.5-turbo", messages=messages)
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"Chat failed: {e}")

    print("\n=== Streaming Chat ===")
    try:
        messages = [
            ChatMessage(role=MessageRole.USER, content="Count from 1 to 5."),
        ]
        for chunk in client.chat_stream(model="gpt-3.5-turbo", messages=messages):
            if chunk.choices and chunk.choices[0].message.content:
                print(chunk.choices[0].message.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"Streaming chat failed: {e}")

    print("\n=== Text Completion ===")
    try:
        response = client.complete(model="gpt-3.5-turbo", prompt="The weather today is")
        print(f"Completion: {response.choices[0].text}")
    except Exception as e:
        print(f"Completion failed: {e}")

    print("\n=== Embeddings ===")
    try:
        response = client.embed(model="text-embedding-ada-002", input="Hello world")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"Embedding failed: {e}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
