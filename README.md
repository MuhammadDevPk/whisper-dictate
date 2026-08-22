# Whisper On-Demand Dictation Utility for macOS

A production-grade, lightweight, real-time voice-to-text dictation utility optimized for macOS. The tool runs persistently in the background, captures microphone audio on a global hotkey toggle, transcribes using `faster-whisper` on the CPU, and automates simulated typing at your active cursor.

Specifically optimized for Intel Core i9 MacBooks to balance speed, RAM usage, and transcription accuracy.

---

## Key Features

* 🧠 **Persistent Model Loading**: Whisper resides permanently in RAM (`float32` CPU execution with 6 calculation threads), avoiding launch latency.
* 🎹 **Global Hotkey Toggle**: Toggle recording globally using `Option + S` or `Cmd + Shift + S`, or press `Enter` in the terminal fallback.
* 🎙️ **Startup Microphone Calibration**: Dynamically calculates your room's ambient noise floor at startup to set the optimal VAD threshold (prevents MacBook fan noise from interfering with speech detection).
* 🟢 **Continuous Dictation & Auto-Commit**: Transcribes sentences in the background and commits them as soon as you pause speaking for `0.8` seconds, clearing the buffer for your next sentence instantly.
* 🛡️ **Pre-Speech Buffer Cushion**: Stores the last `320ms` of audio during silence. When speech is detected, the cushion is prepended, ensuring Whisper never cuts off the beginning of words.
* ⌨️ **Real-Time Diff-Based Typing**: Injects transcribed text character-by-character into any active input box using `pynput`, with dynamic backspace corrections for provisional stream updates.
* 💡 **Smart Newline Formatting**: Appends text on the same line, inserting a newline (`\n`) only if you pause speaking for `3.0` seconds or more.
* 📊 **Compact Terminal UI**: Displays a single, non-wrapping VAD status bar showing real-time RMS, noise floors, accumulated speech duration, and provisional text previews.

---

## Requirements

* **OS**: macOS (10.15 Catalina or newer)
* **Python**: 3.9 - 3.12 (highly recommended to install using `uv`)
* **Libraries**: `pynput`, `sounddevice`, `scipy`, `numpy`, `faster-whisper`
* **macOS Permissions**: Microphone Access, Accessibility, Input Monitoring (if prompted)

---

## Installation

### 1. Initialize Project & Virtual Env
Using `uv` is recommended for fast installation:
```bash
uv venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required packages:
```bash
uv pip install pynput sounddevice scipy numpy faster-whisper
```
*(If you are compiling `whisper.cpp` or using default pip, install using `pip install -r requirements.txt` if available).*

---

## Usage

### 1. Run the Script
Activate your virtual environment and start the utility:
```bash
.venv/bin/python dictate.py
```

### 2. Grant macOS Permissions
When run for the first time, macOS will request access permissions:
1. **Microphone Access**: Required by `sounddevice` to capture speech.
2. **Accessibility**: Required by `pynput` to simulate native system typing.
3. **Input Monitoring**: Required to register background hotkeys globally.

*Tip: If accessibility typing fails, open your macOS System Settings -> Privacy & Security -> Accessibility, and ensure your terminal app (e.g. Terminal, iTerm2, or VS Code) is checked.*

### 3. Start Dictating
1. Position your cursor in any text editor, browser input, or document.
2. Press **Option + S** (or **Cmd + Shift + S**). You will hear a short startup silence calibration (only at script start).
3. The terminal will display `🟢` (speaking status). Start talking.
4. **Important**: Make sure to click focus into your destination text editor so the script types there instead of back into your terminal prompt!
5. Stop speaking for `0.8s` to commit a sentence, or press **Option + S** again to stop recording entirely.

---

## Configuration & Tuning

You can open `dictate.py` and customize several key parameters at the top of the file:

```python
# Whisper Model Configuration
MODEL_SIZE = "small"       # Multilingual model (supports both English and Urdu)
DEVICE = "cpu"
COMPUTE_TYPE = "int8"      # 2-3x faster than float32 on CPU
CPU_THREADS = 6

# Initial prompt to guide Whisper on spelling, context, and proper nouns
INITIAL_PROMPT_EN = None
INITIAL_PROMPT_UR = None

# VAD Settings
VAD_SENSITIVITY = 2.8      # Noise-floor multiplier (lower = more sensitive; range 1.8 - 3.5)
AUTO_COMMIT_DELAY = 1.0    # Seconds of silence before committing a segment
STREAMING_INTERVAL = 0.4   # Seconds between streaming transcription updates
```

### Tuning Recommendations:
* **For Name & Spellings**: Prime `INITIAL_PROMPT_EN` or `INITIAL_PROMPT_UR` with your name and common vocabulary. For example: `INITIAL_PROMPT_EN = "Hello, my name is Muhammad. I am from Sargodha."`
* **VAD Sensitivity**: If the room is very quiet, you can lower `VAD_SENSITIVITY` to `2.2` to capture whispers easily. If you are in a loud room or laptop fans are spinning, raise it to `3.2` to avoid false triggers.
* **Asymmetric Noise Floor Tracking**: The utility implements an asymmetric tracker to dynamically adapt to macOS Automatic Gain Control (AGC) gain-boosting. This tracks background room noise perfectly and prevents VAD from triggering on silence.

---

## Troubleshooting

* **Terminal keeps typing nonsense when dictation starts**: You are focused on the terminal prompt. Switch your focus (click) into another text app like VS Code, Notes, or Chrome.
* **Whisper repeats words/hallucinated text (like "Muhammad, Sargodha")**: Ensure that `INITIAL_PROMPT_EN` and `INITIAL_PROMPT_UR` are set to `None` if they are causing hallucinations on silence or noise. Ensure the active language matches what you are speaking. If you speak English while Urdu (`ur`) is selected, Whisper will hallucinate.
* **Program does not exit**: You can type `exit`, `quit`, or `q` directly in the terminal, or press `Ctrl + C` to exit cleanly.
