#!/usr/bin/env python3

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DASH_DIR = Path(__file__).parent.resolve()
DATA_DIR = (DASH_DIR / ".." / "data").resolve()
OUTPUT_DIR = DASH_DIR / "demo_videos"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "dashboard_demo.mp4"

DATASET_NAME = "UNSW_NB15"          # Cybersecurity dataset matches the UI theme
DEMO_DURATION = 45                  # Seconds of live dashboard footage
TITLE_DURATION = 5                  # Seconds for the title card
VIEWPORT_WIDTH = 1400
VIEWPORT_HEIGHT = 900
VIDEO_FPS = 30

# Narration script (avoid the word "real-time"; use "online" / "continuously" instead)
NARRATION = (
    "Welcome to the Online Classification Simulation Dashboard. "
    "This demo shows the OGC algorithm performing online threat detection "
    "on the UNSW-NB15 network intrusion dataset. "
    "After selecting the dataset, the simulation begins. "
    "The dashboard updates continuously with metrics including false negative rate "
    "and false positive rate, while the system logs each processed chunk. "
    "When a potential threat is detected, an alert appears with the top influential "
    "network features driving the prediction, such as protocol type and source port activity. "
    "The pause and resume controls allow users to inspect the model state "
    "at any point during the stream."
)

# Temporary demo app settings (faster than real-time for visible progress)
CHUNK_SIZE_ORIGINAL = "CHUNK_SIZE = 1"
CHUNK_SIZE_DEMO = "CHUNK_SIZE = 100"
SLEEP_ORIGINAL = "time.sleep(2)"
SLEEP_DEMO = "time.sleep(0.15)"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_demo_app() -> Path:
    """Create a temporary fast-forward copy of app.py."""
    src = DASH_DIR / "app.py"
    dst = DASH_DIR / "app_demo.py"
    content = src.read_text()

    if CHUNK_SIZE_ORIGINAL not in content:
        raise RuntimeError("Could not find CHUNK_SIZE definition in app.py")
    if SLEEP_ORIGINAL not in content:
        raise RuntimeError("Could not find time.sleep(2) in app.py")

    content = content.replace(CHUNK_SIZE_ORIGINAL, CHUNK_SIZE_DEMO)
    content = content.replace(SLEEP_ORIGINAL, SLEEP_DEMO)

    dst.write_text(content)
    return dst


def run_flask(app_module: str) -> subprocess.Popen:
    """Launch the Flask app without debug/reloader."""
    env = os.environ.copy()
    env["FLASK_ENV"] = "production"
    env["FLASK_DEBUG"] = "0"

    cmd = [
        sys.executable,
        "-c",
        f"from {app_module} import app; app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)",
    ]
    return subprocess.Popen(
        cmd,
        cwd=DASH_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_for_server(timeout: int = 30) -> None:
    """Poll the root URL until Flask is serving."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Flask server did not start in time")


def create_title_card(width: int, height: int, duration: int) -> ImageClip:
    """Render a simple title card as a MoviePy ImageClip."""
    img = Image.new("RGB", (width, height), color="#1a253c")
    draw = ImageDraw.Draw(img)

    title = "Online Classification Simulation Dashboard"
    subtitle = "Online OGC threat detection demo"

    # Try to use a nice system font, fall back to default if unavailable
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    def load_font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return None

    title_font = load_font(font_paths[0], 48) or load_font(font_paths[2], 48) or ImageFont.load_default()
    subtitle_font = load_font(font_paths[1], 28) or load_font(font_paths[3], 28) or ImageFont.load_default()

    def text_size(font, text):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    tw, th = text_size(title_font, title)
    sw, sh = text_size(subtitle_font, subtitle)

    draw.text(((width - tw) / 2, height / 2 - th - 20), title, fill="white", font=title_font)
    draw.text(((width - sw) / 2, height / 2 + 20), subtitle, fill="#cccccc", font=subtitle_font)

    card_path = OUTPUT_DIR / "title_card.png"
    img.save(card_path)
    return ImageClip(str(card_path), duration=duration)


def record_dashboard() -> Path:
    """Use Playwright to record the dashboard."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            record_video_dir=str(OUTPUT_DIR),
            record_video_size={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        )
        page = context.new_page()

        # Load dashboard
        page.goto("http://127.0.0.1:5000/", wait_until="networkidle")
        page.wait_for_selector("#dataset-select")
        time.sleep(2)  # Show initial idle state

        # Select dataset
        page.select_option("#dataset-select", DATASET_NAME)
        time.sleep(1)

        # Start simulation
        page.click("#start-button")

        # Let simulation run and update the UI
        time.sleep(DEMO_DURATION)

        # Pause briefly to demonstrate the control, then resume
        page.click("#pause-button")
        time.sleep(2)
        page.click("#pause-button")
        time.sleep(3)

        # Finish
        context.close()
        browser.close()

        return Path(page.video.path())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Building temporary demo app...")
    demo_app_path = build_demo_app()
    demo_module = demo_app_path.stem

    proc = None
    raw_video_path = None

    try:
        print("Starting Flask server...")
        proc = run_flask(demo_module)

        print("Waiting for Flask server...")
        wait_for_server()

        print("Recording dashboard...")
        raw_video_path = record_dashboard()
        print(f"Raw recording: {raw_video_path}")

        print("Adding title card and voiceover...")
        raw_clip = VideoFileClip(str(raw_video_path))
        title_clip = create_title_card(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, TITLE_DURATION)

        final_clip = concatenate_videoclips([title_clip, raw_clip], method="compose")

        # Generate narration audio
        narration_path = OUTPUT_DIR / "narration.mp3"
        tts = gTTS(NARRATION, lang="en", slow=False)
        tts.save(str(narration_path))
        audio_clip = AudioFileClip(str(narration_path))

        # Trim audio if it is longer than the video
        if audio_clip.duration > final_clip.duration:
            audio_clip = audio_clip.subclip(0, final_clip.duration)

        final_clip = final_clip.with_audio(audio_clip)

        print("Exporting final MP4...")
        final_clip.write_videofile(
            str(OUTPUT_PATH),
            codec="libx264",
            audio_codec="aac",
            fps=VIDEO_FPS,
        )

        final_clip.close()
        raw_clip.close()
        title_clip.close()
        audio_clip.close()

        print(f"Done. Final video saved to: {OUTPUT_PATH}")

    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Clean up temporary files
        if demo_app_path.exists():
            demo_app_path.unlink()
        if raw_video_path and raw_video_path.exists():
            raw_video_path.unlink()
        title_card_path = OUTPUT_DIR / "title_card.png"
        if title_card_path.exists():
            title_card_path.unlink()
        narration_path = OUTPUT_DIR / "narration.mp3"
        if narration_path.exists():
            narration_path.unlink()


if __name__ == "__main__":
    main()
