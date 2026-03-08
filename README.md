# 🎬 Auto Transcriber

An intelligent video-to-text transcription pipeline that automatically extracts audio from online videos and generates high-quality transcripts using OpenAI's Whisper model.

## ✨ Features

- **One-Click Transcription**: Simply paste a video URL and get clean transcripts
- **Multi-Platform Support**: Works with YouTube, Bilibili, and 1000+ sites (via yt-dlp)
- **Dual-File Output**: Saves both audio files (WAV) and clean transcripts (TXT)
- **GPU Acceleration**: CUDA support for NVIDIA GPUs (tested on RTX 5060 Laptop)
- **Smart Language Detection**: Automatically detects video language
- **Web Interface**: Clean Gradio UI for easy usage
- **Performance Optimized**: Tuned for faster transcription with minimal quality loss

## 🚀 Tech Stack

- **yt-dlp**: Audio extraction from video URLs
- **OpenAI Whisper**: State-of-the-art speech recognition
- **Gradio 6.0**: Modern web interface
- **PyTorch**: Deep learning framework with CUDA support
- **FFmpeg**: Audio processing and format conversion

## 💻 Hardware Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10 or higher (3.12 recommended for CUDA)
- **RAM**: 8GB+
- **Storage**: 5GB+ for models

### Recommended for GPU Acceleration
- **GPU**: NVIDIA GPU with 8GB+ VRAM
  - ✅ **RTX 5060 Laptop** (Blackwell architecture - requires PyTorch nightly)
  - ✅ RTX 40 series (Ada Lovelace)
  - ✅ RTX 30 series (Ampere)
  - ✅ GTX 16 series (Turing)
- **Drivers**: Latest NVIDIA GPU drivers

### Special Note for RTX 5060 Users
The RTX 5060 (Blackwell architecture, sm_120) requires **PyTorch nightly build with cu128** support:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 📦 Installation

### ⚠️ Prerequisites (Important!)

**Before installing, please ensure you have the following:**

#### 1. Python 3.12 (Required)
The Windows launcher (`run.bat`) expects Python 3.12 at the default path:
- `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python312\python.exe`

**How to install Python 3.12 correctly:**
1. Download Python 3.12 from [python.org](https://www.python.org/downloads/release/python-3120/)
2. During installation, **check "Add Python to PATH"**
3. Install to the default location (do not change the path)
4. Verify installation: `py -3.12 --version`

**If you have a different Python version:**
- Python 3.13 is **not compatible** with PyTorch CUDA
- Python 3.10-3.11 may work but 3.12 is recommended
- If Python is installed elsewhere, edit `run.bat` and update the `PYTHON_CMD` path

#### 2. FFmpeg (Required)
**The launcher will automatically detect FFmpeg in these locations:**
- System PATH (if added manually)
- WinGet default path: `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\`

**Recommended installation method (Windows):**
```bash
winget install ffmpeg
```

**Alternative (Manual):**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to system PATH

**Verify installation:**
```bash
ffmpeg -version
```

#### 3. CUDA GPU Drivers (Optional, for GPU acceleration)
- **NVIDIA GPU**: Latest drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
- **RTX 5060 Users**: Requires PyTorch nightly build (see below)

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/auto-transcriber.git
cd auto-transcriber
```

### Step 2: Install PyTorch (Choose One)

**For CPU-only usage:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA GPU acceleration (RTX 40 series and older):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For RTX 5060 (Blackwell architecture) - Requires nightly build:**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 3: Install Python Dependencies
```bash
pip install gradio yt-dlp openai-whisper
```

**Or install all at once:**
```bash
pip install -r requirements.txt
```

**Note**: The `run.bat` launcher will automatically install dependencies on first run, but manual installation is recommended for better control.

**Windows (WinGet)**:
```bash
winget install ffmpeg
```

**Windows (Manual)**:
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to system PATH

**Linux/macOS**:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 🎯 Usage

### Running the Application

**Windows**:
```bash
# Double-click run.bat
# OR
run.bat
```

**Linux/macOS**:
```bash
python app.py
```

The web interface will automatically open at `http://127.0.0.1:7860`

### Using the Web Interface

1. Paste a video URL (YouTube, Bilibili, etc.)
2. Select language (or leave as "Auto Detect")
3. Enable VAD filter to remove silence
4. Click "Start Transcription"
5. Wait for processing
6. Find output files in the `Output_Workspace` folder

### Command Line Usage

You can also use the core library directly in your Python scripts:

```python
from transcriber_core import VideoTranscriberPipeline

# Initialize pipeline
pipeline = VideoTranscriberPipeline(
    model_size="large-v3",
    device="cuda"  # or "cpu"
)

# Process video
result = pipeline.process(
    url="https://www.youtube.com/watch?v=xxxxx",
    language="auto",  # auto-detect language
    vad_filter=True
)

# Access results
print(f"Transcript: {result.transcript_text}")
print(f"Audio saved to: {result.audio_path}")
print(f"Transcript saved to: {result.transcript_path}")
```

## 📁 Output Files

All output files are saved in the `Output_Workspace` directory:

```
Output_Workspace/
├── [Video_Title_ID].wav    # Audio file (16kHz, mono, PCM16)
└── [Video_Title_ID].txt    # Clean transcript (no timestamps)
```

**Example**:
- `How_to_Learn_Python_abc123.wav` - Extracted audio
- `How_to_Learn_Python_abc123.txt` - Clean transcript text

## ⚡ Performance Optimizations

This implementation includes several optimizations for faster transcription:

- **Temperature=0.0**: Faster, more deterministic decoding
- **FP16 for CUDA**: Half-precision floating point for GPU inference
- **Optimized thresholds**: Tuned compression ratio and no-speech thresholds
- **Disabled context tracking**: Faster processing for long videos

**Expected Performance** (RTX 5060 Laptop, large-v3 model):
- ~1-2x faster than real-time (5-minute video in ~2.5-5 minutes)

## 🔧 Troubleshooting

### Python 3.12 Not Found or Wrong Version

**Error**: `Python 3.12 not found at expected location`

**Solutions**:

1. **Install Python 3.12 to default location**:
   ```bash
   # Download from: https://www.python.org/downloads/release/python-3120/
   # Install with default settings and "Add to PATH" checked
   ```

2. **Edit run.bat if Python is installed elsewhere**:
   - Open `run.bat` in a text editor
   - Find line: `set "PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"`
   - Change to your Python path, e.g.:
   ```batch
   set "PYTHON_CMD=C:\Python312\python.exe"
   ```

3. **Verify Python version**:
   ```bash
   py -3.12 --version
   # Should output: Python 3.12.x
   ```

4. **Common mistakes**:
   - ❌ Using Python 3.13 (not compatible with PyTorch CUDA)
   - ❌ Installing Python to custom path without updating run.bat
   - ❌ Forgetting to check "Add Python to PATH" during installation

---

### FFmpeg Not Found

**Error**: `FFmpeg not found`

**Solutions**:

1. **Install via WinGet (Recommended)**:
   ```bash
   winget install ffmpeg
   ```

2. **Verify installation**:
   ```bash
   ffmpeg -version
   ```

3. **Manual installation**:
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to system PATH:
     - Press Win+R, type `sysdm.cpl`
     - Go to "Advanced" → "Environment Variables"
     - Edit "Path" variable
     - Add `C:\ffmpeg\bin`

4. **WinGet installation detection**:
   - The launcher automatically detects FFmpeg at:
   - `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-*-full_build\bin\`
   - If using WinGet, no PATH configuration needed

### CUDA Not Available
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install PyTorch with CUDA support
# For RTX 40 series and older:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For RTX 5060:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Model Download Fails
```bash
# Set Hugging Face mirror (China users)
set HF_ENDPOINT=https://hf-mirror.com
python app.py
```

### RTX 5060 CUDA Compatibility Error
**Error**: `CUDA capability sm_120 is not compatible with current PyTorch`

**Solution**: Install PyTorch nightly build with cu128 support:
```bash
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Out of Memory
- Close other GPU-intensive applications
- Use a smaller model: `model_size="medium"` or `model_size="small"`
- Restart the application to clear GPU memory

## 📂 Project Structure

```
auto-transcriber/
├── app.py                      # Gradio web interface
├── transcriber_core.py         # Core transcription pipeline
├── run.bat                     # Windows launcher (auto-detects dependencies)
├── install_dependencies.bat    # Manual dependency installer (optional)
├── requirements.txt            # Python dependencies list
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── Output_Workspace/           # Output files (auto-created)
```

**Clean and minimal**: Only 6 core files needed for full functionality!

## 🎨 Architecture

```
┌─────────────┐
│  Video URL  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   yt-dlp        │  → Extract audio as WAV (16kHz, mono)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  OpenAI Whisper │  → Transcribe with CUDA/CPU
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Output Files   │  → [Title_ID].wav + [Title_ID].txt
└─────────────────┘
```

## 🔬 Technical Details

### Audio Processing
- **Format**: WAV (PCM16)
- **Sample Rate**: 16kHz (Whisper native)
- **Channels**: Mono
- **Bit Depth**: 16-bit

### Transcription Model
- **Model**: OpenAI Whisper Large-V3
- **Parameters**: 1.5B parameters
- **Languages**: 99 languages
- **Accuracy**: State-of-the-art speech recognition

### Supported Video Sites
- YouTube
- Bilibili
- Vimeo
- Twitter/X
- TikTok
- And 1000+ more sites via yt-dlp

## 🤝 Contributing

This project was developed using **GLM vibe coding** - an AI-assisted development approach.

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is for personal use only.

## 🙏 Acknowledgments

- **OpenAI** for the Whisper model
- **yt-dlp** team for the video downloader
- **Gradio** team for the web UI framework
- **PyTorch** team for the deep learning framework
- **Hugging Face** for model hosting

## 📄 Development Method

This project was developed with assistance from **GLM vibe coding** - an AI-powered collaborative development approach that combines human creativity with AI efficiency to accelerate software development.

---

**Version**: 1.0.0
**Last Updated**: March 2026
**Status**: ✅ Production Ready
