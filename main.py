"""Motivational Video Creator FastAPI service."""

from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import List

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip
from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Environment configuration
# ----------------------------------------------------------------------------

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENABLE_OPENAI_SPEND = os.getenv("ENABLE_OPENAI_SPEND", "false").lower() == "true"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/motivational-video-api"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Application bootstrap
# ----------------------------------------------------------------------------

app = FastAPI(
    title="Motivational Video Creator API",
    description="Generate short inspirational quote videos backed by Pexels stock footage.",
    version="1.0.0",
)

# ----------------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    topic: str = Field(..., description="The motivational theme or niche for the quotes.")
    quotes: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of quotes to include in the video (1-5).",
    )


class GenerateResponse(BaseModel):
    video_path: str
    quotes: List[str]


# ----------------------------------------------------------------------------
# Quote helpers
# ----------------------------------------------------------------------------

CURATED_QUOTES = {
    "motivation": [
        "Believe in yourself, and anything is possible.",
        "Every step forward is a step toward success.",
        "Dream it. Plan it. Do it.",
        "Your potential is endless.",
        "Small progress is still progress.",
    ],
    "self-belief": [
        "You are stronger than your doubts.",
        "Confidence is built one brave moment at a time.",
        "Trust the process and trust yourself.",
        "Your belief fuels your journey.",
        "Shine as bright as you imagine.",
    ],
    "resilience": [
        "Setbacks are setups for comebacks.",
        "Storms make trees take deeper roots.",
        "Rise every time you fall.",
        "Keep moving; your breakthrough is near.",
        "Strength grows in the moments you think you can't go on but keep going anyway.",
    ],
}


def get_openai_client() -> OpenAI | None:
    if not ENABLE_OPENAI_SPEND:
        return None
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenAI spending enabled but OPENAI_API_KEY is missing.",
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_quote(topic: str) -> str:
    """Generate a single quote using OpenAI or fall back to curated content."""

    client = get_openai_client()
    if client is None:
        return random.choice(CURATED_QUOTES.get(topic.lower(), CURATED_QUOTES["motivation"]))

    prompt = (
        "Craft a short motivational quote (max 20 words) about "
        f"'{topic}'. Respond with JSON: {{\"quote\": \"...\"}}"
    )

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
    except Exception as exc:  # pragma: no cover - network failure path
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc

    try:
        content = response.output[0].content[0].text
        data = json.loads(content)
        quote = data["quote"].strip().strip("\"")
    except (KeyError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=500, detail="Unexpected OpenAI response payload.") from exc

    return quote


def gather_quotes(topic: str, count: int) -> List[str]:
    quotes = []
    for _ in range(count):
        quotes.append(generate_quote(topic))
    return quotes


# ----------------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------------


def fetch_random_video(topic: str) -> Path:
    if not PEXELS_API_KEY:
        raise HTTPException(status_code=400, detail="PEXELS_API_KEY is required to fetch videos.")

    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": topic, "per_page": 15}

    try:
        response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=15)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach Pexels: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Pexels responded with status {response.status_code}.")

    payload = response.json()
    videos = payload.get("videos") or []
    if not videos:
        raise HTTPException(status_code=404, detail="No videos found for the requested topic.")

    chosen = random.choice(videos)
    video_files = chosen.get("video_files") or []
    if not video_files:
        raise HTTPException(status_code=404, detail="Chosen Pexels video has no downloadable files.")

    # Prefer HD files if available, otherwise fall back to the first entry.
    video_files.sort(key=lambda item: item.get("quality") == "hd", reverse=True)
    video_url = video_files[0]["link"]

    try:
        download_response = requests.get(video_url, timeout=30)
        download_response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download video asset: {exc}") from exc

    temp_name = f"raw-{uuid.uuid4().hex}.mp4"
    video_path = OUTPUT_DIR / temp_name
    video_path.write_bytes(download_response.content)
    return video_path


def create_text_overlay(text: str, size: tuple[int, int], duration: float, start: float) -> ImageClip:
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Add semi-transparent background rectangle for readability.
    overlay_height = int(height * 0.35)
    overlay_top = (height - overlay_height) // 2
    draw.rectangle(
        [(int(width * 0.1), overlay_top), (int(width * 0.9), overlay_top + overlay_height)],
        fill=(0, 0, 0, 160),
    )

    # Word-wrap the text.
    max_width = int(width * 0.75)
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    for word in words:
        trial_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), trial_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    text_block = "\n".join(lines)
    text_bbox = draw.multiline_textbbox((0, 0), text_block, font=font, align="center")
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = overlay_top + (overlay_height - text_height) // 2

    draw.multiline_text((text_x, text_y), text_block, fill=(255, 255, 255, 255), font=font, align="center")

    frame = np.array(image)
    clip = ImageClip(frame).set_duration(duration).set_start(start)
    return clip


def compose_video(topic: str, quotes: List[str]) -> Path:
    raw_video_path = fetch_random_video(topic)
    clip = VideoFileClip(str(raw_video_path))

    target_duration = min(clip.duration, 45)
    clip = clip.subclip(0, target_duration)

    overlays = []
    segment_duration = target_duration / len(quotes)
    for index, quote in enumerate(quotes):
        start_time = index * segment_duration
        overlays.append(create_text_overlay(quote, clip.size, segment_duration, start_time))

    final_clip = CompositeVideoClip([clip, *overlays])

    output_name = f"motivational-{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_DIR / output_name
    final_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=min(int(clip.fps) if clip.fps else 24, 30),
        verbose=False,
        logger=None,
    )

    clip.close()
    final_clip.close()
    try:
        raw_video_path.unlink(missing_ok=True)
    except AttributeError:  # Python < 3.8 compatibility
        if raw_video_path.exists():
            raw_video_path.unlink()

    return output_path


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_video(request: GenerateRequest) -> GenerateResponse:
    quotes = gather_quotes(request.topic, request.quotes)
    try:
        video_path = compose_video(request.topic, quotes)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime failures from moviepy/ffmpeg
        raise HTTPException(status_code=500, detail=f"Video composition failed: {exc}") from exc

    return GenerateResponse(video_path=str(video_path), quotes=quotes)


@app.get("/download")
def download_video(path: str = Query(..., description="Absolute path returned from the /generate endpoint.")) -> FileResponse:
    resolved_path = Path(path)
    try:
        resolved_path.relative_to(OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Can only download files generated by this service.")

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Requested file does not exist.")

    return FileResponse(resolved_path)


# ----------------------------------------------------------------------------
# Entrypoint for local execution
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
