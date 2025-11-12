import os
import requests
import random
from openai import OpenAI
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Load API keys from environment variables
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Output folder for videos
OUTPUT_DIR = "/app/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_random_video(query="nature"):
    """Fetch a random video from Pexels."""
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 10}
    res = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)
    data = res.json()
    videos = data.get("videos", [])
    if not videos:
        raise Exception("No videos found for query.")
    chosen = random.choice(videos)
    video_url = chosen["video_files"][0]["link"]
    video_path = os.path.join(OUTPUT_DIR, "clip.mp4")
    print(f"Downloading video from: {video_url}")
    with open(video_path, "wb") as f:
        f.write(requests.get(video_url).content)
    return video_path

def generate_caption(niche="motivation"):
    """Generate a motivational caption using OpenAI."""
    prompt = f"Write a short motivational quote for a {niche} video."
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    text = response.output[0].content[0].text.strip('"')
    print(f"Generated caption: {text}")
    return text

def make_video():
    print("\n🎬 Starting video creation...")
    video_path = get_random_video("motivation")
    caption = generate_caption()

    clip = VideoFileClip(video_path)
    clip = clip.subclip(0, min(15, clip.duration))

    # Overlay caption text
    text = TextClip(caption, fontsize=50, color='white', size=clip.size)
    text = text.set_duration(clip.duration).set_position('center')

    final = CompositeVideoClip([clip, text])
    output_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
    final.write_videofile(output_path, codec="libx264", fps=24)
    print(f"✅ Video saved at: {output_path}\n")

if __name__ == "__main__":
    make_video()
