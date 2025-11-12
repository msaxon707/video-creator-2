import json
import os
import textwrap
import uuid
from pathlib import Path
from typing import List

import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, VideoFileClip
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

APP_NAME = "motivational-video-api"
PEXELS_VIDEO_ENDPOINT = "https://api.pexels.com/videos/search"
PEXELS_AUDIO_ENDPOINT = "https://api.pexels.com/audio/search"
DEFAULT_TOPICS = [
    "perseverance",
    "self-belief",
    "discipline",
    "growth mindset",
    "courage",
]

app = FastAPI(title="Motivational Video Creator API")


class GenerateRequest(BaseModel):
    topic: str = Field(
        default="perseverance",
        description="General theme or keyword for sourcing media and quotes.",
    )
    quotes: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Number of motivational quotes to overlay in the video.",
    )


class GeneratedAsset(BaseModel):
    video_path: str
    quotes: List[str]


class OpenAISafetyGate:
    """Utility that prevents accidental OpenAI spend without explicit opt-in."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.allow_spend = os.getenv("ENABLE_OPENAI_SPEND", "false").lower() == "true"
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_quotes(self, topic: str, count: int) -> List[str]:
        if self.allow_spend and self.client:
            prompt = textwrap.dedent(
                f"""
                You are an inspirational coach. Create {count} concise motivational quotes
                about {topic}. Each quote must be unique, 12 to 22 words long, and avoid
                quotation marks. Return the quotes as a JSON array of strings.
                """
            ).strip()
            response = self.client.responses.create(
                model="gpt-4o",
                input=prompt,
                max_output_tokens=600,
            )
            try:
                result_text = response.output_text  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - defensive
                raise HTTPException(500, "OpenAI client missing output_text field")
            try:
                data = json.loads(result_text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise HTTPException(500, f"OpenAI response was not JSON: {exc}")
            if not isinstance(data, list):
                raise HTTPException(500, "OpenAI response JSON was not a list")
            return [str(q).strip() for q in data][:count]
        # Fallback curated quotes to guarantee zero cost without explicit opt-in
        curated = [
            "Small daily victories build the momentum for extraordinary change.",
            "Discomfort is the evidence that you are stretching into your potential.",
            "Energy follows intention—set it high and your actions will follow.",
            "Courage grows louder every time you listen to it.",
            "Discipline is the quiet engine that drives impossible goals into reality.",
            "Progress loves consistency more than perfection.",
        ]
        matches = [quote for quote in curated if topic.lower() in quote.lower()]
        pool = matches or curated
        repeated = (pool * ((count // len(pool)) + 1))[:count]
        return repeated


safety_gate = OpenAISafetyGate()


def _ensure_dirs() -> Path:
    base_dir = Path(os.getenv("OUTPUT_DIR", "/tmp")) / APP_NAME
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _download_file(url: str, suffix: str) -> Path:
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise HTTPException(502, f"Failed to download media from {url}")
    dest = _ensure_dirs() / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(response.content)
    return dest


def _fetch_pexels_resource(endpoint: str, params: dict) -> dict:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        raise HTTPException(400, "PEXELS_API_KEY is required")
    headers = {"Authorization": api_key}
    response = requests.get(endpoint, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        raise HTTPException(502, f"Pexels API error: {response.text}")
    return response.json()


def _fetch_video(topic: str) -> Path:
    payload = _fetch_pexels_resource(
        PEXELS_VIDEO_ENDPOINT, {"query": topic or DEFAULT_TOPICS[0], "per_page": 5}
    )
    videos = payload.get("videos", [])
    if not videos:
        raise HTTPException(404, "No videos found for topic")
    chosen = videos[0]
    video_files = chosen.get("video_files", [])
    if not video_files:
        raise HTTPException(404, "Selected video missing files")
    # Prefer mp4 1080p, otherwise first available
    sorted_files = sorted(
        video_files,
        key=lambda item: (
            0 if item.get("quality") == "hd" else 1,
            item.get("width", 0) * item.get("height", 0),
        ),
    )
    url = sorted_files[0]["link"]
    return _download_file(url, ".mp4")


def _fetch_audio(topic: str) -> Path:
    payload = _fetch_pexels_resource(
        PEXELS_AUDIO_ENDPOINT,
        {"query": topic or "motivation", "per_page": 10},
    )
    tracks = payload.get("audio", [])
    if not tracks:
        raise HTTPException(404, "No audio tracks found")
    best = max(tracks, key=lambda item: item.get("duration", 0))
    audio_files = best.get("audio_files", [])
    if not audio_files:
        raise HTTPException(404, "Selected audio track missing files")
    preferred = sorted(
        audio_files,
        key=lambda item: (
            0 if item.get("file_type") == "mp3" else 1,
            -item.get("bitrate", 0),
        ),
    )
    return _download_file(preferred[0]["link"], ".mp3")


def _wrap_text(text: str, width: int = 34) -> str:
    lines = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)


def _build_text_clip(quotes: List[str], size: tuple[int, int], duration: float) -> ImageClip:
    width, height = size
    wrapped_blocks = [_wrap_text(quote) for quote in quotes]
    padding = 60
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    translucent = Image.new("RGBA", (width, height), (0, 0, 0, 130))
    canvas = Image.alpha_composite(canvas, translucent)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=48)
    except OSError:  # pragma: no cover - environment fallback
        font = ImageFont.load_default()
    font_size = getattr(font, "size", 32)
    y = padding
    for paragraph in "\n".join(wrapped_blocks + [""]).split("\n"):
        if not paragraph:
            y += int(font_size * 0.8)
            continue
        bbox = draw.textbbox((0, 0), paragraph, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        x = (width - line_width) // 2
        draw.text((x, y), paragraph, font=font, fill=(255, 255, 255, 235))
        y += line_height + 6
    array = np.array(canvas)
    clip = ImageClip(array).set_duration(duration)
    clip = clip.set_opacity(0.9)
    return clip


def _compose_video(video_path: Path, audio_path: Path, quotes: List[str]) -> Path:
    output_path = _ensure_dirs() / f"{uuid.uuid4().hex}.mp4"
    with VideoFileClip(str(video_path)) as video_clip:
        duration = video_clip.duration
        text_clip = _build_text_clip(quotes, video_clip.size, duration)
        text_clip = text_clip.set_position(("center", "center"))
        with AudioFileClip(str(audio_path)) as audio_clip:
            trimmed_audio = audio_clip.subclip(0, min(audio_clip.duration, duration))
            composite = CompositeVideoClip([video_clip.set_audio(trimmed_audio), text_clip])
            composite.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                fps=video_clip.fps,
                temp_audiofile=str(_ensure_dirs() / f"temp-{uuid.uuid4().hex}.m4a"),
                remove_temp=True,
            )
    return output_path


def _cleanup_files(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/generate", response_model=GeneratedAsset)
def generate_video(payload: GenerateRequest, background_tasks: BackgroundTasks) -> GeneratedAsset:
    topic = payload.topic.strip() or DEFAULT_TOPICS[0]
    quotes = safety_gate.generate_quotes(topic, payload.quotes)
    video_path = _fetch_video(topic)
    audio_path = _fetch_audio(topic)
    output_path = _compose_video(video_path, audio_path, quotes)
    background_tasks.add_task(_cleanup_files, video_path, audio_path)
    return GeneratedAsset(video_path=str(output_path), quotes=quotes)


@app.get("/download")
def download_video(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(str(file_path), media_type="video/mp4", filename=file_path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
