import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import torch
import whisper

try:
    import spaces

    ZERO_GPU_AVAILABLE = True
except ImportError:
    ZERO_GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatsAppConverter:
    _model = None  # Class variable for model caching

    def __init__(self, whisper_model: str = "base"):
        """Initialize the converter with specified Whisper model."""
        if WhatsAppConverter._model is None:
            WhatsAppConverter._model = whisper.load_model(whisper_model)
        self.model = WhatsAppConverter._model

        # Set batch size based on environment
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            if "A100" in gpu_name:
                self.batch_size = 8
            elif "T4" in gpu_name:
                self.batch_size = 4
            else:
                self.batch_size = 3
        else:
            self.batch_size = 2

        # Support multiple language patterns for attachments
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

    @spaces.GPU
    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file using Whisper with automatic language detection."""
        try:
            # Configure transcription options
            transcribe_options = {}
            if torch.cuda.is_available():
                transcribe_options["fp16"] = True

            # Load and transcribe audio
            audio = whisper.load_audio(audio_path)
            result = self.model.transcribe(audio, **transcribe_options)

            detected_language = result["language"]
            logger.info(f"Detected language: {detected_language}")
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    @spaces.GPU
    def batch_transcribe(self, wav_paths: List[Tuple[str, str]]) -> Dict[str, str]:
        """Transcribe a batch of audio files efficiently."""
        results = {}

        try:
            if torch.cuda.is_available() and len(wav_paths) > 1:
                # Load all audio files in batch
                audio_batch = [
                    whisper.load_audio(wav_path) for _, wav_path in wav_paths
                ]

                # Transcribe batch with GPU optimization
                transcribe_options = {"fp16": True, "batch_size": len(audio_batch)}
                batch_results = self.model.transcribe(audio_batch, **transcribe_options)

                # Store results
                for (opus_name, _), result in zip(wav_paths, batch_results):
                    results[opus_name] = result["text"].strip()
            else:
                # Process individually
                for opus_name, wav_path in wav_paths:
                    transcription = self.transcribe_audio(wav_path)
                    if transcription:
                        results[opus_name] = transcription

        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            # Fall back to individual processing
            for opus_name, wav_path in wav_paths:
                transcription = self.transcribe_audio(wav_path)
                if transcription:
                    results[opus_name] = transcription

        return results

    def process_chat_file(self, chat_path: str, audio_files: dict) -> str:
        """Process chat file and replace audio references with transcriptions."""
        try:
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
        except Exception as e:
            logger.error(f"Chat file processing failed: {e}")
            return None

    def process_zip(self, zip_path: str, progress_callback=None) -> Tuple[str, str]:
        """Process WhatsApp chat export ZIP file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract ZIP
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find chat file and audio files
                chat_file = None
                opus_files = []

                for file in Path(temp_dir).rglob("*"):
                    if file.name == "_chat.txt":
                        chat_file = str(file)
                    elif file.suffix.lower() == ".opus":
                        opus_files.append(file)

                if not chat_file:
                    return "Error: No _chat.txt found in ZIP file.", None

                total_files = len(opus_files)
                processed_files = 0
                audio_files = {}

                # Process audio files in optimized batches
                for i in range(0, total_files, self.batch_size):
                    batch = opus_files[i : i + self.batch_size]
                    wav_paths = []

                    # Convert batch to WAV in parallel
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        wav_futures = [
                            executor.submit(
                                self.convert_opus_to_wav,
                                str(file),
                                str(file.with_suffix(".wav")),
                            )
                            for file in batch
                        ]
                        for file, future in zip(batch, wav_futures):
                            if future.result():
                                wav_paths.append(
                                    (file.name, str(file.with_suffix(".wav")))
                                )

                    # Process batch
                    batch_results = self.batch_transcribe(wav_paths)
                    audio_files.update(batch_results)

                    processed_files += len(batch)
                    if progress_callback:
                        progress = (processed_files / total_files) * 100
                        progress_callback(progress)

                # Process chat file
                processed_content = self.process_chat_file(chat_file, audio_files)
                if not processed_content:
                    return "Error: Failed to process chat file.", None

                # Save output
                output_path = os.path.join(temp_dir, "processed_chat.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(processed_content)

                # Copy to a permanent location
                final_output = f"processed_chat_{Path(zip_path).stem}.txt"
                shutil.copy2(output_path, final_output)

                return "Success: Chat processing complete.", final_output

            except Exception as e:
                logger.error(f"ZIP processing failed: {e}")
                return f"Error: Failed to process ZIP file: {str(e)}", None


def create_ui() -> gr.Interface:
    """Create Gradio interface."""

    def process_file(
        zip_file: str, model_name: str, progress: gr.Progress = None
    ) -> Tuple[str, str]:
        converter = WhatsAppConverter(whisper_model=model_name)
        return converter.process_zip(zip_file, progress)

    description = """Upload a WhatsApp chat export ZIP file to convert audio messages to text.
    Model selection guide:
    - tiny: Fast but less accurate
    - base: Good balance for most uses
    - small: Better accuracy
    - medium: High accuracy
    - large: Best accuracy"""

    if ZERO_GPU_AVAILABLE:
        description += "\n\nNote: This Space uses ZeroGPU for dynamic GPU allocation."

    interface = gr.Interface(
        fn=process_file,
        inputs=[
            gr.File(label="Upload WhatsApp Chat Export (ZIP)"),
            gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="base",
                label="Whisper Model",
                info="Larger models are more accurate but slower",
            ),
        ],
        outputs=[gr.Textbox(label="Status"), gr.File(label="Processed Chat File")],
        title="WhatsApp Chat to LLM-Ready Text Converter",
        description=description,
    )
    return interface


if __name__ == "__main__":
    # Check Python version
    import sys

    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        exit(1)

    # Check FFmpeg installation
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH")
        exit(1)

    # Launch Gradio interface
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
