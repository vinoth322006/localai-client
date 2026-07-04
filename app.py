import json
from flask import Flask, render_template, request, Response, jsonify
import httpx

app = Flask(__name__)

LOCALAI_BASE_URL = "http://localhost:20128/v1"
LOCALAI_API_KEY = "sk-5de6a8c7de457cda-f63afc-642571a9"


def get_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LOCALAI_API_KEY}",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "base_url": LOCALAI_BASE_URL,
        "has_api_key": bool(LOCALAI_API_KEY),
    })


@app.route("/api/models")
def list_models():
    try:
        resp = httpx.get(
            f"{LOCALAI_BASE_URL}/models",
            headers=get_headers(),
            timeout=10,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    model = data.get("model", "")
    stream = data.get("stream", False)
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 2048)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if stream:
        def generate():
            with httpx.Client(timeout=120) as client:
                with client.stream(
                    "POST",
                    f"{LOCALAI_BASE_URL}/chat/completions",
                    json=payload,
                    headers=get_headers(),
                ) as resp:
                    for line in resp.iter_lines():
                        if line and line.startswith("data: "):
                            yield line + "\n\n"
                            if line.strip() == "data: [DONE]":
                                break

        return Response(generate(), mimetype="text/event-stream")

    try:
        resp = httpx.post(
            f"{LOCALAI_BASE_URL}/chat/completions",
            json=payload,
            headers=get_headers(),
            timeout=120,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/complete", methods=["POST"])
def complete():
    data = request.json
    payload = {
        "model": data.get("model", ""),
        "prompt": data.get("prompt", ""),
        "temperature": data.get("temperature", 0.7),
        "max_tokens": data.get("max_tokens", 256),
        "stream": False,
    }
    try:
        resp = httpx.post(
            f"{LOCALAI_BASE_URL}/completions",
            json=payload,
            headers=get_headers(),
            timeout=120,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.json
    payload = {
        "model": data.get("model", ""),
        "input": data.get("input", ""),
    }
    try:
        resp = httpx.post(
            f"{LOCALAI_BASE_URL}/embeddings",
            json=payload,
            headers=get_headers(),
            timeout=60,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    try:
        resp = httpx.get(
            f"{LOCALAI_BASE_URL}/../health",
            headers=get_headers(),
            timeout=5,
        )
        return jsonify(resp.json())
    except Exception:
        try:
            resp = httpx.get(
                f"{LOCALAI_BASE_URL}/models",
                headers=get_headers(),
                timeout=5,
            )
            if resp.status_code == 200:
                return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
