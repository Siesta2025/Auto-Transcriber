#!/usr/bin/env python3
"""
Video Transcriber Core - Backend pipeline for Gradio UI
Windows-optimized with dual-file output preservation
Using OpenAI Whisper (better CUDA compatibility)
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import yt_dlp
import whisper
import torch


# Configure Hugging Face cache (for compatibility)
DEFAULT_HF_CACHE = os.path.expanduser("~/.cache/whisper")
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


@dataclass
class TranscriptionResult:
    """Data class for transcription results."""
    success: bool
    message: str
    workspace_path: str
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    transcript_text: Optional[str] = None
    video_title: Optional[str] = None
    video_id: Optional[str] = None
    error: Optional[str] = None


class WorkspaceManager:
    """Manage output workspace directory and file naming."""

    def __init__(self, workspace_name: str = "Output_Workspace"):
        """
        Initialize workspace manager.

        Args:
            workspace_name: Name of the workspace directory
        """
        self.workspace_name = workspace_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_workspace_path(self) -> Path:
        """Get or create workspace path in script directory."""
        # Get script directory (works for .py and frozen exe)
        if getattr(sys, 'frozen', False):
            script_dir = Path(sys.executable).parent
        else:
            script_dir = Path(__file__).parent.resolve()

        workspace_path = script_dir / self.workspace_name
        workspace_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Workspace path: {workspace_path}")
        return workspace_path

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for Windows (remove invalid characters).

        Args:
            filename: Raw filename string

        Returns:
            Sanitized filename safe for Windows
        """
        # Windows invalid characters: < > : " / \ | ? *
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        # Limit length (Windows MAX_PATH is 260, but keep reasonable)
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        return sanitized or "output"

    def generate_output_names(self, video_title: str, video_id: str) -> Tuple[str, str, str]:
        """
        Generate output filenames for audio and transcript.

        Args:
            video_title: Video title
            video_id: Video ID

        Returns:
            Tuple of (base_name, audio_filename, transcript_filename)
        """
        # Sanitize title for Windows filename
        safe_title = self.sanitize_filename(video_title)
        base_name = f"{safe_title}_{video_id}"

        audio_filename = f"{base_name}.wav"
        transcript_filename = f"{base_name}.txt"

        return base_name, audio_filename, transcript_filename


class AudioExtractor:
    """Extract audio from video URLs using yt-dlp Python API."""

    def __init__(self):
        """Initialize AudioExtractor."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_audio(self, url: str, output_path: Path,
                     output_filename: str) -> Tuple[str, str, str]:
        """
        Extract audio from video URL and save as WAV.

        Args:
            url: Video URL
            output_path: Output directory path
            output_filename: Output filename

        Returns:
            Tuple of (audio_filepath, video_title, video_id)

        Raises:
            Exception: If extraction fails
        """
        output_template = str(output_path / output_filename.replace('.wav', ''))

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ac', '1',           # Mono channel
                '-ar', '16000',        # 16kHz sample rate
                '-acodec', 'pcm_s16le' # PCM 16-bit little-endian
            ],
            'quiet': True,
            'no_warnings': True,
        }

        try:
            self.logger.info(f"Extracting audio from: {url}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get title and ID
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown_Title')
                video_id = info.get('id', info.get('display_id', 'unknown'))

                # Download and process
                ydl.download([url])

            # Find the actual output file
            audio_path = self._find_output_file(output_template)
            if not audio_path or not os.path.exists(audio_path):
                raise FileNotFoundError(f"Output file not found: {output_template}")

            self.logger.info(f"Audio extracted: {audio_path}")
            return audio_path, video_title, video_id

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise

    def _find_output_file(self, base_path: str) -> Optional[str]:
        """Find the actual output file."""
        if os.path.exists(base_path):
            return base_path

        base_without_ext = base_path.rsplit('.', 1)[0]
        for ext in ['.wav', '.m4a', '.mp3', '.webm']:
            candidate = base_without_ext + ext
            if os.path.exists(candidate):
                return candidate
        return None


class Transcriber:
    """Transcribe audio using OpenAI Whisper with CUDA support."""

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        """
        Initialize Transcriber with OpenAI Whisper.

        Args:
            model_size: Whisper model size (large-v3 for best quality)
            device: Device to use (cuda/cpu)
        """
        self.model_size = model_size
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

        # Map model_size to Whisper's naming
        self.model_name_map = {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large-v1": "large",
            "large-v2": "large",
            "large-v3": "large",  # OpenAI Whisper uses "large" for all v1/v2/v3
        }

    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            model_name = self.model_name_map.get(self.model_size, "large")
            self.logger.info(f"Loading Whisper model: {model_name}")
            self.logger.info(f"Requested device: {self.device}")

            # Check CUDA availability
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"

            # Load model (OpenAI Whisper loads to CPU by default)
            try:
                self.model = whisper.load_model(model_name)
                self.logger.info(f"Model loaded, moving to {self.device}...")

                # Move to correct device
                if self.device == "cuda":
                    self.model = self.model.to(torch.device("cuda"))
                    self.logger.info("Model moved to CUDA successfully")
                else:
                    self.logger.info("Model on CPU")

            except Exception as e:
                if self.device == "cuda":
                    self.logger.warning(f"Failed to use CUDA: {e}")
                    self.logger.info("Falling back to CPU...")
                    self.device = "cpu"
                    self.model = whisper.load_model(model_name)
                    self.logger.info("Model loaded on CPU")
                else:
                    raise

            self.logger.info("Model loading complete")

    def transcribe(self, audio_path: str, language: str = "auto") -> str:
        """
        Transcribe audio file to clean text.

        Args:
            audio_path: Path to WAV audio file
            language: Language code (None for auto-detection)

        Returns:
            Clean transcript text (without timestamps)

        Raises:
            Exception: If transcription fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None:
            self.load_model()

        try:
            self.logger.info(f"Starting transcription: {audio_path}")
            self.logger.info(f"Using device for inference: {self.device}")

            # Language parameter: None for auto-detect
            lang_param = None if language == "auto" else language

            # For CUDA, explicitly move model to GPU before transcribe
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(torch.device("cuda"))
                self.logger.info("Model moved to CUDA device")

            # Transcribe with performance optimizations
            # temperature=0: Faster, more deterministic decoding
            # compression_ratio_threshold=2.4: Skip compression checks
            # no_speech_threshold=0.1: Lower threshold to process more audio
            # condition_on_previous_text=False: Disable context for speed
            result = self.model.transcribe(
                audio_path,
                language=lang_param,
                task="transcribe",
                verbose=False,
                fp16=(self.device == "cuda"),  # Use fp16 only for CUDA
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.1,
                condition_on_previous_text=False
            )

            # Build clean transcript (no timestamps or segments)
            transcript_text = result["text"].strip()

            detected_lang = result.get("language", language)

            self.logger.info(f"Transcription completed. Language: {detected_lang}")
            self.logger.info(f"Total characters: {len(transcript_text)}")

            return transcript_text

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise


class VideoTranscriberPipeline:
    """Complete pipeline: URL -> Audio -> Transcript (dual-file preservation)."""

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        """
        Initialize the complete pipeline.

        Args:
            model_size: Whisper model size
            device: Device for inference (cuda/cpu)
        """
        self.model_size = model_size
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.workspace = WorkspaceManager()
        self.extractor = AudioExtractor()
        self.transcriber = Transcriber(
            model_size=model_size,
            device=device
        )

    def process(self, url: str, language: str = "auto",
                vad_filter: bool = True, min_silence_duration_ms: int = 500) -> TranscriptionResult:
        """
        Process video URL to generate transcript and preserve audio.

        Args:
            url: Video URL
            language: Language code
            vad_filter: Enable VAD filtering (ignored in OpenAI Whisper)
            min_silence_duration_ms: Minimum silence duration (ignored in OpenAI Whisper)

        Returns:
            TranscriptionResult with all output information
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING PIPELINE")
            self.logger.info("=" * 60)

            # Prepare workspace
            workspace_path = self.workspace.get_workspace_path()

            # Step 1: Extract audio with temp name first
            self.logger.info("STEP 1: Audio Extraction")
            temp_filename = "temp_audio.wav"
            audio_path, video_title, video_id = self.extractor.extract_audio(
                url, workspace_path, temp_filename
            )

            # Step 2: Generate proper filenames and rename
            self.logger.info("STEP 2: File Organization")
            base_name, audio_filename, transcript_filename = \
                self.workspace.generate_output_names(video_title, video_id)

            # Rename audio file to final name
            final_audio_path = workspace_path / audio_filename
            if os.path.exists(audio_path):
                if os.path.exists(final_audio_path):
                    os.remove(final_audio_path)
                os.rename(audio_path, str(final_audio_path))
                self.logger.info(f"Renamed audio to: {final_audio_path}")

            # Step 3: Transcribe
            self.logger.info("STEP 3: Transcription")
            transcript_text = self.transcriber.transcribe(
                str(final_audio_path),
                language=language
            )

            # Step 4: Save transcript
            self.logger.info("STEP 4: Saving Transcript")
            transcript_path = workspace_path / transcript_filename
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)

            self.logger.info(f"Transcript saved: {transcript_path}")

            # Return success result
            return TranscriptionResult(
                success=True,
                message="Processing completed successfully!",
                workspace_path=str(workspace_path),
                audio_path=str(final_audio_path),
                transcript_path=str(transcript_path),
                transcript_text=transcript_text,
                video_title=video_title,
                video_id=video_id
            )

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(
                success=False,
                message="Processing failed",
                workspace_path=str(self.workspace.get_workspace_path()),
                error=str(e)
            )


# Import sys for frozen detection
import sys
