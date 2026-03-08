#!/usr/bin/env python3
"""
Video Transcriber - Gradio Web UI
Windows-optimized interface for video-to-text transcription
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from transcriber_core import (
    VideoTranscriberPipeline,
    TranscriptionResult
)


# Configure logging
def setup_logging():
    """Configure logging for clean console output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Suppress verbose logging from libraries
    logging.getLogger('yt_dlp').setLevel(logging.ERROR)
    logging.getLogger('faster_whisper').setLevel(logging.WARNING)


class TranscriberUI:
    """Gradio UI for Video Transcriber."""

    def __init__(self):
        """Initialize the UI components."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipeline: Optional[VideoTranscriberPipeline] = None

    def initialize_pipeline(self):
        """Initialize the transcription pipeline (lazy loading)."""
        if self.pipeline is None:
            self.logger.info("Initializing pipeline...")

            # Check if CUDA is available
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use smaller model for CPU
            model_size = "large-v3" if device == "cuda" else "medium"

            self.logger.info(f"Using device: {device}, model: {model_size}")

            self.pipeline = VideoTranscriberPipeline(
                model_size=model_size,
                device=device
            )

    def process_url(
        self,
        url: str,
        language: str,
        vad_filter: bool,
        min_silence_ms: int,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Process video URL and return results for UI.

        Args:
            url: Video URL
            language: Language selection
            vad_filter: Enable VAD filter
            min_silence_ms: Minimum silence duration
            progress: Gradio progress tracker

        Returns:
            Tuple of (status_message, file_info, transcript_text)
        """
        self.logger.info(f"Processing URL: {url}")

        if not url or not url.strip():
            error_msg = "❌ Error: Please enter a valid URL"
            self.logger.warning(error_msg)
            return error_msg, "", ""

        # Initialize pipeline if needed
        try:
            self.logger.info("Initializing pipeline...")
            self.initialize_pipeline()
        except Exception as e:
            error_msg = f"❌ Failed to initialize pipeline:\n\n{str(e)}"
            self.logger.error(error_msg)
            return error_msg, "", ""

        # Process with progress tracking
        try:
            progress(0.1, desc="Extracting audio...")

            self.logger.info("Calling pipeline.process()...")
            result: TranscriptionResult = self.pipeline.process(
                url=url,
                language=language if language != "Auto-detect" else "auto",
                vad_filter=vad_filter,
                min_silence_duration_ms=min_silence_ms
            )

            self.logger.info(f"Pipeline result: success={result.success}")

            if result.success:
                # Build success message
                status_msg = f"""✅ Success!

📺 Video: {result.video_title}
🆔 ID: {result.video_id}
🌐 Language: {language if language != 'Auto-detect' else 'Auto-detected'}

Files saved to Output_Workspace folder."""

                # Build file info
                file_info = f"""📁 Workspace: {result.workspace_path}

🎵 Audio File:
{Path(result.audio_path).name}

📄 Transcript File:
{Path(result.transcript_path).name}"""

                progress(1.0, desc="Completed!")
                self.logger.info("Transcription completed successfully")
                return status_msg, file_info, result.transcript_text or ""

            else:
                error_msg = f"❌ Processing Failed\n\n{result.error or 'Unknown error'}"
                progress(1.0, desc="Failed")
                self.logger.error(f"Processing failed: {result.error}")
                return error_msg, "", ""

        except KeyboardInterrupt:
            error_msg = "❌ Operation cancelled by user"
            self.logger.warning(error_msg)
            return error_msg, "", ""

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"❌ Unexpected Error:\n\n{str(e)}\n\nCheck console for details."
            self.logger.error(f"UI processing error:\n{error_details}")
            return error_msg, "", ""

    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the Gradio interface.

        Returns:
            Configured Gradio Blocks interface
        """
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 900px !important;
            margin: auto !important;
        }
        .status-box {
            font-family: Consolas, 'Courier New', monospace;
            white-space: pre-wrap;
        }
        footer {
            display: none !important;
        }
        """

        with gr.Blocks(
            title="Video Transcriber",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            # Header
            gr.Markdown(
                """
                # 🎬 Video Transcriber
                ### URL → Clean Audio → Pure Transcript (Local AI, No API Fees)
                """
            )

            gr.Markdown("""**Supports:** YouTube, Bilibili | **Model:** Whisper Large-V3 (int8) | **GPU:** RTX 5060 8GB""")

            gr.Markdown("---")

            # Input Section
            with gr.Row():
                with gr.Column(scale=3):
                    url_input = gr.Textbox(
                        label="📹 Video URL",
                        placeholder="https://www.youtube.com/watch?v=... or https://www.bilibili.com/video/BV...",
                        lines=2,
                        autofocus=True
                    )

                with gr.Column(scale=1):
                    language = gr.Dropdown(
                        choices=["Auto-detect", "zh", "en", "ja", "ko", "es", "fr", "de", "ru"],
                        value="Auto-detect",
                        label="🌐 Language"
                    )

            # Advanced Options (collapsed)
            with gr.Accordion("⚙️ Advanced Options", open=False):
                with gr.Row():
                    vad_filter = gr.Checkbox(
                        value=True,
                        label="Enable VAD Filter (removes background silence)"
                    )
                    min_silence_ms = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=100,
                        label="Min Silence Duration (ms)"
                    )

            # Process Button
            process_btn = gr.Button(
                "🚀 Start Transcription",
                variant="primary",
                size="lg"
            )

            gr.Markdown("---")

            # Output Section
            gr.Markdown("### 📊 Output")

            with gr.Row():
                with gr.Column(scale=1):
                    status_output = gr.Textbox(
                        label="🔔 Status",
                        lines=6,
                        interactive=False,
                        elem_classes=["status-box"]
                    )

                with gr.Column(scale=1):
                    file_output = gr.Textbox(
                        label="📂 Saved Files",
                        lines=6,
                        interactive=False,
                        elem_classes=["status-box"]
                    )

            # Transcript Preview
            transcript_output = gr.Textbox(
                label="📝 Transcript Preview",
                lines=15,
                interactive=False,
                placeholder="Transcript will appear here after processing..."
            )

            # Wire up the button
            process_btn.click(
                fn=self.process_url,
                inputs=[url_input, language, vad_filter, min_silence_ms],
                outputs=[status_output, file_output, transcript_output]
            )

            # Examples
            gr.Markdown("---")
            gr.Examples(
                examples=[
                    ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "Auto-detect", True, 500],
                ],
                inputs=[url_input, language, vad_filter, min_silence_ms],
                label="📋 Example (click to fill)"
            )

            # Footer
            gr.Markdown(
                """
                ---
                **Output Workspace:** `Output_Workspace/` folder in the same directory as this script

                Both `.wav` (audio) and `.txt` (transcript) files are preserved.
                """
            )

        return interface


def main():
    """Main entry point for the Gradio application."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Video Transcriber Web UI...")

    try:
        # Create UI
        ui = TranscriberUI()
        app = ui.create_interface()

        # Launch with browser auto-open
        logger.info("Launching Gradio interface...")
        app.launch(
            inbrowser=True,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            quiet=False
        )

    except KeyboardInterrupt:
        logger.info("Application stopped by user")

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================
#
# Install required packages:
#
#   pip install gradio yt-dlp faster-whisper
#
# For Windows, ensure FFmpeg is installed and in system PATH:
#
#   - Download from: https://ffmpeg.org/download.html
#   - Or use winget: winget install ffmpeg
#   - Add FFmpeg bin folder to system PATH
#
# ============================================================================
# MODEL DOWNLOAD (First Run Only)
# ============================================================================
#
# On first run, the Whisper Large-V3 model (~3GB) will be downloaded
# automatically from Hugging Face Hub.
#
# If download fails, set mirror for China users:
#
#   set HF_ENDPOINT=https://hf-mirror.com
#   python app.py
#
# ============================================================================
