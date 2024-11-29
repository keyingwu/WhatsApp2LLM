import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatsAppConverter:
    _model = None  # Class variable for model caching

    def __init__(
        self, whisper_model: str = "base"
    ):  # Changed to tiny for better performance
        """Initialize the converter with specified Whisper model."""
        if WhatsAppConverter._model is None:
            WhatsAppConverter._model = whisper.load_model(whisper_model)
        self.model = WhatsAppConverter._model
        self.attachment_patterns = [
            r"<附件：(.+?)>",  # Chinese
            r"<attached: (.+?)>",  # English
            r"<attachment: (.+?)>",  # Alternative English
            r"<Anhang: (.+?)>",  # German
            r"<archivo adjunto: (.+?)>",  # Spanish
            r"<pièce jointe : (.+?)>",  # French
            r"<allegato: (.+?)>",  # Italian
            r"<anexo: (.+?)>",  # Portuguese
            r"<添付ファイル：(.+?)>",  # Japanese
            r"<첨부파일: (.+?)>",  # Korean
        ]

    def convert_opus_to_wav(self, opus_path: str, wav_path: str) -> bool:
        """Convert opus file to wav format using FFmpeg."""
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    opus_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    wav_path,
                    "-y",
                    "-loglevel",
                    "error",
                ],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            return False

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file using Whisper with automatic language detection."""
        try:
            result = self.model.transcribe(audio_path)
            detected_language = result["language"]
            logger.info(f"Detected language: {detected_language}")
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def process_chat_file(self, chat_path: str, audio_files: dict) -> str:
        """Process chat file and replace audio references with transcriptions."""
        with open(chat_path, "r", encoding="utf-8") as f:
            content = f.read()

        for pattern in self.attachment_patterns:

            def replace_match(match):
                audio_file = match.group(1)
                if audio_file in audio_files:
                    return f"[Audio Transcription: {audio_files[audio_file]}]"
                return match.group(0)

            content = re.sub(pattern, replace_match, content)

        return content

    def process_zip(self, zip_path: str, progress_callback=None) -> Tuple[str, str]:
        """Process WhatsApp chat export ZIP file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find chat file and audio files
            chat_file = None
            audio_files = {}
            opus_files = []

            for file in Path(temp_dir).rglob("*"):
                if file.name == "_chat.txt":
                    chat_file = str(file)
                elif file.suffix.lower() == ".opus":
                    opus_files.append(file)

            total_files = len(opus_files)
            processed_files = 0

            # Process audio files in batches
            batch_size = 3  # Adjust based on your machine's capabilities
            for i in range(0, total_files, batch_size):
                batch = opus_files[i : i + batch_size]
                wav_paths = []

                # Convert all files in batch to WAV
                for file in batch:
                    wav_path = str(file.with_suffix(".wav"))
                    if self.convert_opus_to_wav(str(file), wav_path):
                        wav_paths.append((file.name, wav_path))

                # Process all WAV files in batch
                for opus_name, wav_path in wav_paths:
                    transcription = self.transcribe_audio(wav_path)
                    if transcription:
                        audio_files[opus_name] = transcription

                    processed_files += 1
                    if progress_callback:
                        progress = (processed_files / total_files) * 100
                        progress_callback(progress)

            if not chat_file:
                return "Error: No _chat.txt found in ZIP file.", None

            # Process audio files
            for file in Path(temp_dir).rglob("*.opus"):
                wav_path = str(file.with_suffix(".wav"))

                if self.convert_opus_to_wav(str(file), wav_path):
                    transcription = self.transcribe_audio(wav_path)
                    if transcription:
                        audio_files[file.name] = transcription

                processed_files += 1
                if progress_callback:
                    progress = (processed_files / total_files) * 100
                    progress_callback(progress)

            # Process chat file
            processed_content = self.process_chat_file(chat_file, audio_files)

            # Save output
            output_path = os.path.join(temp_dir, "processed_chat.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(processed_content)

            # Copy to a permanent location
            final_output = f"processed_chat_{Path(zip_path).stem}.txt"
            shutil.copy2(output_path, final_output)

            return "Success: Chat processing complete.", final_output


def create_ui() -> gr.Interface:
    """Create Gradio interface."""

    def process_file(
        zip_file: str, model_name: str, progress: gr.Progress
    ) -> Tuple[str, str]:
        converter = WhatsAppConverter(whisper_model=model_name)
        return converter.process_zip(zip_file, progress)

    interface = gr.Interface(
        fn=process_file,
        inputs=[
            gr.File(label="Upload WhatsApp Chat Export (ZIP)"),
            gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="tiny",
                label="Whisper Model",
                info="Larger models are more accurate but slower. tiny: fastest, large: most accurate",
            ),
        ],
        outputs=[gr.Textbox(label="Status"), gr.File(label="Processed Chat File")],
        title="WhatsApp Chat to LLM-Ready Text Converter",
        description="""Upload a WhatsApp chat export ZIP file to convert audio messages to text.
        Model selection guide:
        - tiny (1GB VRAM): Fast but less accurate
        - base (1GB VRAM): Good balance for most uses
        - small (2GB VRAM): Better accuracy
        - medium (5GB VRAM): High accuracy
        - large (10GB VRAM): Best accuracy""",
    )
    return interface


if __name__ == "__main__":
    # Check FFmpeg installation
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH")
        exit(1)

    # Launch Gradio interface
    app = create_ui()
    app.launch()
