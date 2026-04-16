# pip install youtube-transcript-api yt-dlp openai

# export OPENAI_API_KEY="твой_ключ"
# python yt_to_md.py "https://www.youtube.com/watch?v=VIDEO_ID"

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp


def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from common URL formats.
    Supports:
    - https://www.youtube.com/watch?v=...
    - https://youtu.be/...
    - https://www.youtube.com/shorts/...
    """
    parsed = urlparse(url)

    if parsed.netloc in {"youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0]

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [None])[0]
            if video_id:
                return video_id

        if parsed.path.startswith("/shorts/"):
            parts = parsed.path.split("/")
            if len(parts) >= 3 and parts[2]:
                return parts[2]

        if parsed.path.startswith("/embed/"):
            parts = parsed.path.split("/")
            if len(parts) >= 3 and parts[2]:
                return parts[2]

    raise ValueError(f"Не удалось извлечь video_id из URL: {url}")


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[<>:\"/\\\\|?*\\x00-\\x1F]", "_", name)
    return name[:180].strip() or "transcript"


def format_ts(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_video_title(url: str) -> str:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("title") or "YouTube video"


def try_get_youtube_transcript(video_id: str, languages: list[str] | None = None):
    """
    Try to fetch transcript from YouTube directly.
    Returns list[dict] with keys: text, start, duration
    """
    languages = languages or ["ru", "en"]
    return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)


def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download best audio and convert to mp3 using ffmpeg.
    """
    output_template = str(output_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": False,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    result = output_dir / f"{video_id}.mp3"
    if not result.exists():
        raise FileNotFoundError(f"Аудиофайл не найден после загрузки: {result}")
    return result


def transcribe_with_openai(audio_path: Path, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    Cloud transcription via OpenAI.
    Requires OPENAI_API_KEY in environment.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with audio_path.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=f,
        )

    # In current Python SDK examples this object usually exposes `.text`
    text = getattr(transcript, "text", None)
    if not text:
        raise RuntimeError("OpenAI не вернул текст транскрибации")
    return text


def transcript_segments_to_markdown(
    title: str,
    url: str,
    segments: list[dict],
    source_label: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- URL: {url}",
        f"- Source: {source_label}",
        "",
        "## Transcript",
        "",
    ]

    for seg in segments:
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start") or 0)
        if text:
            lines.append(f"[{format_ts(start)}] {text}")

    lines.append("")
    return "\n".join(lines)


def plain_text_to_markdown(title: str, url: str, text: str, source_label: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- URL: {url}",
            f"- Source: {source_label}",
            "",
            "## Transcript",
            "",
            text.strip(),
            "",
        ]
    )


def main():
    if len(sys.argv) < 2:
        print("Использование: python yt_to_md.py <youtube_url> [output_dir]")
        sys.exit(1)

    url = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(url)
    title = get_video_title(url)
    md_name = sanitize_filename(title) + ".md"
    md_path = out_dir / md_name

    # 1. Try direct transcript first
    try:
        segments = try_get_youtube_transcript(video_id, languages=["ru", "en"])
        markdown = transcript_segments_to_markdown(
            title=title,
            url=url,
            segments=segments,
            source_label="YouTube transcript",
        )
        md_path.write_text(markdown, encoding="utf-8")
        print(f"Готово: {md_path}")
        return
    except (TranscriptsDisabled, NoTranscriptFound):
        print("Субтитры YouTube не найдены, перехожу к ASR...")
    except Exception as e:
        print(f"Не удалось получить YouTube transcript: {e}")
        print("Перехожу к скачиванию аудио и ASR...")

    # 2. Fallback: download audio + transcribe
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio_path = download_audio(url, tmp_dir)
        text = transcribe_with_openai(audio_path)
        markdown = plain_text_to_markdown(
            title=title,
            url=url,
            text=text,
            source_label="OpenAI transcription",
        )
        md_path.write_text(markdown, encoding="utf-8")
        print(f"Готово: {md_path}")


if __name__ == "__main__":
    main()