# Motivational Video Creator API

This project exposes a FastAPI microservice that composes motivational quote videos by blending GPT-generated (or curated) quotes with royalty-free stock video and audio from Pexels.

The deployment footprint has been limited to just three files so it works seamlessly with Coolify or similar minimal build pipelines:

1. `main.py` – application entrypoint with the FastAPI routes and media pipeline.
2. `requirements.txt` – Python dependencies for the service.
3. `README.md` – usage instructions and configuration details.

## Safety features

OpenAI usage is opt-in. By default the service **does not** call OpenAI APIs, preventing accidental spend. To permit calls to GPT-4o you must explicitly set `ENABLE_OPENAI_SPEND=true`. When that flag is absent, the service falls back to a curated set of motivational quotes.

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `PEXELS_API_KEY` | ✅ | Pexels API key for video and audio sourcing. |
| `OPENAI_API_KEY` | ⚠️ | Required only if you want GPT-4o to craft bespoke quotes. |
| `ENABLE_OPENAI_SPEND` | ⚠️ | Must be set to `true` (lowercase) to allow OpenAI spending. Defaults to `false`. |
| `OUTPUT_DIR` | ❌ | Optional directory for generated assets. Defaults to `/tmp`. |
| `PORT` | ❌ | Port for the uvicorn server. Defaults to `8000`. |

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PEXELS_API_KEY=your_key ENABLE_OPENAI_SPEND=false uvicorn main:app --reload
```

## API

### `POST /generate`

**Request body**

```json
{
  "topic": "self-belief",
  "quotes": 3
}
```

**Response body**

```json
{
  "video_path": "/tmp/motivational-video-api/abc123.mp4",
  "quotes": ["…", "…"]
}
```

The `video_path` references a local file on the server. You can download it via the `/download` route.

### `GET /download?path=...`

Streams the generated `mp4` file back to the client.

## Docker deployment

The included `Dockerfile` targets Python 3.11 on Debian slim and installs the necessary `ffmpeg` binary for MoviePy. Build and run the container locally with:

```bash
docker build -t motivational-video-api .
docker run --rm -p 8000:8000 \
  -e PEXELS_API_KEY=your_key \
  -e ENABLE_OPENAI_SPEND=false \
  motivational-video-api
```

Override the port by setting the `PORT` environment variable.

## Running on Coolify

You can deploy either via the plain Python buildpack or by selecting the Dockerfile in your service configuration. For the former, ensure the build command installs dependencies (`pip install -r requirements.txt`) and the start command launches `uvicorn main:app --host 0.0.0.0 --port $PORT`.

Set the required environment variables in Coolify secrets before deploying. If you opt into the Dockerfile, Coolify will automatically expose port `8000` unless overridden by the `PORT` environment variable.
