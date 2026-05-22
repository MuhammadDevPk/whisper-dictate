#!/usr/bin/env python3
"""
dictate.py - On-Demand Continuous Voice-to-Text Dictation Utility

Requirements:
1. Persistent accelerated base.en Whisper model loaded strictly on CPU (6 threads, float32).
2. Global Keyboard Hook (Option + S or Cmd + Shift + S) or Enter to toggle continuous dictation.
3. Sound capture buffer via sounddevice InputStream at 16kHz, mono channel.
4. Voice Activity Detection (VAD) using startup microphone noise calibration.
5. Real-time streaming transcription running every 1.0s with diff-based typing and backspace correction.
6. Smart Newline logic: appends text on same line unless user pauses for >= 3.0 seconds.
7. Clean terminal status line to prevent scrolling clutter.
"""

import sys
import time
import queue
import threading
import traceback
import numpy as np
import sounddevice as sd
from pynput import keyboard
from faster_whisper import WhisperModel

# Terminal colors for clean logging
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Whisper Model Configuration
# Note: For significantly higher accuracy, you can change this to "small.en" or "small" (multilingual).
# If the Intel CPU handles it fine, you can even try "medium.en".
MODEL_SIZE = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
CPU_THREADS = 6

# Initial prompt to guide Whisper on spelling, context, and proper nouns (e.g. "Muhammad", "Sargodha")
# This is a powerful way to guide Whisper's spelling of names and format dictation.
INITIAL_PROMPT = "Hello, my name is Muhammad. I am from Sargodha. I am dictating clear English speech."

# VAD Settings
AUTO_COMMIT_DELAY = 0.8  # Commit audio segment after 0.8 seconds of silence
STREAMING_INTERVAL = 1.0  # Transcribe provisional text every 1.0s of continuous speech
VAD_SENSITIVITY_MULTIPLIER = 1.8  # Calibration noise floor multiplier (Default: 1.8, range 1.5 - 2.5)

# State Machine States
STATE_IDLE = "IDLE"
STATE_RECORDING = "RECORDING"
STATE_TRANSCRIBING = "TRANSCRIBING"

# Global state variables
current_state = STATE_IDLE
state_lock = threading.Lock()

raw_audio_queue = queue.Queue()
transcription_queue = queue.Queue()
audio_stream = None

# Calibrated Threshold (will be configured at startup)
VAD_THRESHOLD = 0.003
latest_transcription = ""

# pynput Keyboard Controller
keyboard_controller = keyboard.Controller()


def audio_callback(indata, frames, time_info, status):
    """Callback function for sounddevice to capture audio chunks."""
    if status:
        print(f"\n{YELLOW}⚠️  Audio stream warning: {status}{RESET}", file=sys.stderr)
    raw_audio_queue.put((indata.copy(), False))


def push_streaming_audio(audio_data, prepend_newline):
    """
    Clears any old/stale streaming chunks from the transcription queue
    to prevent queue lag, keeping final commits and shutdown commands intact.
    """
    temp = []
    while not transcription_queue.empty():
        try:
            item = transcription_queue.get_nowait()
            if item is None or item[1]:  # Keep shutdown signal (None) or final commits (True)
                temp.append(item)
        except queue.Empty:
            break
            
    for item in temp:
        transcription_queue.put(item)
        
    transcription_queue.put((audio_data, False, prepend_newline))


def audio_processing_worker():
    """
    Dedicated worker thread that processes raw audio chunks from raw_audio_queue.
    Performs calibrated threshold VAD filtering and groups active speech frames.
    Commits completed segments and updates provisional streaming chunks.
    """
    global latest_transcription
    speech_buffer = []
    pre_speech_buffer = []
    has_speech = False
    silence_duration = 0.0
    last_stream_time = 0.0
    last_commit_time = 0.0
    should_prepend_newline = False
    
    last_print_time = 0.0
    last_has_speech = False
    
    while True:
        try:
            item = raw_audio_queue.get()
            if item is None:
                raw_audio_queue.task_done()
                break
                
            chunk, is_flush = item
            
            # 1. Handle Flush Signal (recording stopped)
            if is_flush:
                if speech_buffer:
                    audio_data = np.concatenate(speech_buffer, axis=0).flatten()
                    rms = np.sqrt(np.mean(audio_data**2))
                    if rms >= 0.0001:
                        transcription_queue.put((audio_data, True, should_prepend_newline))
                    speech_buffer = []
                has_speech = False
                silence_duration = 0.0
                pre_speech_buffer.clear()
                raw_audio_queue.task_done()
                continue
                
            # 2. Process Standard Audio Chunk
            rms = np.sqrt(np.mean(chunk**2))
            chunk_duration = len(chunk) / SAMPLE_RATE
            
            if rms > VAD_THRESHOLD:
                # Speech starts
                if not has_speech:
                    has_speech = True
                    # Prepend pre-speech buffer to catch the onset of the word
                    speech_buffer = list(pre_speech_buffer)
                    pre_speech_buffer.clear()
                    
                    last_stream_time = time.time()
                    
                    # Smart Newline: Prepend a newline if the pause was 3.0s or more
                    if last_commit_time > 0.0 and (time.time() - last_commit_time >= 3.0):
                        should_prepend_newline = True
                    else:
                        should_prepend_newline = False
                
                speech_buffer.append(chunk)
                silence_duration = 0.0
                
                # Periodically stream the accumulated buffer to Whisper for real-time feedback
                current_time = time.time()
                if current_time - last_stream_time >= STREAMING_INTERVAL:
                    audio_data = np.concatenate(speech_buffer, axis=0).flatten()
                    push_streaming_audio(audio_data, should_prepend_newline)
                    last_stream_time = current_time
            else:
                if has_speech:
                    speech_buffer.append(chunk)
                    silence_duration += chunk_duration
                    
                    # If silence duration exceeds the commit delay, finalize the segment
                    if silence_duration >= AUTO_COMMIT_DELAY:
                        audio_data = np.concatenate(speech_buffer, axis=0).flatten()
                        transcription_queue.put((audio_data, True, should_prepend_newline))
                        
                        last_commit_time = time.time()
                        
                        # Reset for next speech segment
                        speech_buffer = []
                        has_speech = False
                        silence_duration = 0.0
                        pre_speech_buffer.clear()
                else:
                    # Discard quiet chunks during long silence but keep them in pre_speech_buffer
                    pre_speech_buffer.append(chunk)
                    # Limit pre-speech buffer to last ~320ms (5 chunks * 64ms)
                    if len(pre_speech_buffer) > 5:
                        pre_speech_buffer.pop(0)
            
            # Throttle visual real-time status bar updates to ~10Hz or state changes
            current_time = time.time()
            if (current_time - last_print_time >= 0.1) or (has_speech != last_has_speech):
                buf_duration_sec = len(speech_buffer) * chunk_duration
                status_text = f"{GREEN}🟢{RESET}" if has_speech else f"{YELLOW}⚪{RESET}"
                display_text = latest_transcription
                if len(display_text) > 20:
                    display_text = "..." + display_text[-17:]
                
                # Ultra-compact single-line overwrite that fits in standard 80-column terminal
                print(f"\rVAD: {status_text} | RMS: {rms:.4f}/{VAD_THRESHOLD:.4f} | Buf: {buf_duration_sec:.1f}s | \"{CYAN}{display_text}{RESET}\"    ", end="", flush=True)
                last_print_time = current_time
                last_has_speech = has_speech
            
            raw_audio_queue.task_done()
            
        except Exception as e:
            print(f"\n{RED}❌ Error in audio processing thread: {e}{RESET}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def update_typing(old_text, new_text):
    """
    Dynamically typing diffs: compares old_text and new_text,
    sends Backspaces for modified letters, and types new letters.
    """
    common_len = 0
    min_len = min(len(old_text), len(new_text))
    for i in range(min_len):
        if old_text[i] == new_text[i]:
            common_len += 1
        else:
            break
            
    backspaces = len(old_text) - common_len
    suffix_to_type = new_text[common_len:]
    
    # 1. Backspace deleted letters
    if backspaces > 0:
        for _ in range(backspaces):
            keyboard_controller.press(keyboard.Key.backspace)
            keyboard_controller.release(keyboard.Key.backspace)
            time.sleep(0.005)  # 5ms delay to prevent OS dropouts
            
    # 2. Type new letters
    if suffix_to_type:
        keyboard_controller.type(suffix_to_type)


def transcription_worker(model):
    """
    Background worker thread that transcribes committed and streaming segments,
    and updates the active typing buffer.
    """
    global current_state, latest_transcription
    typed_text = ""
    
    while True:
        item = transcription_queue.get()
        if item is None:
            transcription_queue.task_done()
            break
            
        audio_data, is_final, prepend_newline = item
        
        try:
            segments, info = model.transcribe(
                audio_data,
                beam_size=5,
                language="en",
                vad_filter=True,
                initial_prompt=INITIAL_PROMPT,
            )
            text_segments = [seg.text for seg in segments]
            full_text = "".join(text_segments).strip()
            
            if full_text:
                # Prepend a newline exactly once at start of segment if user paused for >= 3s
                if prepend_newline and not typed_text:
                    keyboard_controller.type("\n")
                
                # Perform the differential typing
                update_typing(typed_text, full_text)
                
                if is_final:
                    # Append a trailing space for next segment and clear state
                    keyboard_controller.type(" ")
                    typed_text = ""
                    latest_transcription = ""
                    # Print completed line cleanly in terminal
                    print(f"\n✨ {GREEN}{BOLD}[Committed]{RESET} \"{full_text}\"")
                else:
                    latest_transcription = full_text
                    typed_text = full_text
            else:
                if is_final:
                    # If empty final text, make sure we clear any provisional typing
                    update_typing(typed_text, "")
                    typed_text = ""
                    latest_transcription = ""
                    
        except Exception as e:
            print(f"\n{RED}❌ Error in transcription worker: {e}{RESET}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            transcription_queue.task_done()
            
            # Check if we should transition back to IDLE
            with state_lock:
                if current_state == STATE_TRANSCRIBING and transcription_queue.empty() and audio_stream is None:
                    current_state = STATE_IDLE
                    print(f"\n[Ready] Sitting in background (Press {CYAN}Option+S{RESET}, {CYAN}Cmd+Shift+S{RESET}, or {CYAN}Enter{RESET} to toggle)...")


def toggle_recording():
    """Toggles state between IDLE and RECORDING, spawning worker tasks as appropriate."""
    global current_state, audio_stream
    
    with state_lock:
        if current_state == STATE_IDLE:
            # Transition IDLE -> RECORDING
            while not raw_audio_queue.empty():
                try:
                    raw_audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            try:
                audio_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=1024,
                    callback=audio_callback
                )
                audio_stream.start()
                current_state = STATE_RECORDING
                print(f"\n🟢 {GREEN}{BOLD}Listening...{RESET} (Continuous dictation active. Press Option+S, Cmd+Shift+S, or Enter in terminal to stop)")
            except Exception as e:
                print(f"\n{RED}❌ Failed to start recording: {e}{RESET}", file=sys.stderr)
                print(f"{YELLOW}💡 Tip: Ensure terminal has Microphone Access in macOS System Settings -> Privacy & Security -> Microphone.{RESET}", file=sys.stderr)
                
        elif current_state == STATE_RECORDING:
            # Transition RECORDING -> TRANSCRIBING
            current_state = STATE_TRANSCRIBING
            print(f"\n⏹️  {BLUE}Stopping continuous dictation...{RESET}")
            
            # Stop the audio stream immediately
            if audio_stream:
                try:
                    audio_stream.stop()
                    audio_stream.close()
                except Exception as e:
                    print(f"{RED}❌ Error stopping audio stream: {e}{RESET}", file=sys.stderr)
                audio_stream = None
                
            # Signal the processing worker to flush whatever is remaining in its buffer
            raw_audio_queue.put((None, True))
            
        elif current_state == STATE_TRANSCRIBING:
            print(f"\n{YELLOW}⚠️  Transcribing/Typing in progress. Please wait...{RESET}", end="", flush=True)


def on_hotkey_pressed():
    """Wrapper called when a toggle trigger is fired."""
    threading.Thread(target=toggle_recording, daemon=True).start()


def main():
    global VAD_THRESHOLD
    print(f"{BLUE}{BOLD}=== Whisper On-Demand Dictation Utility ==={RESET}")
    
    # 1. Query audio input device
    try:
        input_device = sd.query_devices(kind="input")
        device_name = input_device.get("name", "Default Device")
        print(f"🎤 Default Input Device: {CYAN}{device_name}{RESET}")
    except Exception as e:
        print(f"{RED}❌ No input audio devices found or sounddevice error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    # 2. Calibrate background noise level
    print(f"🎙️  {CYAN}Calibrating microphone noise floor... Please remain silent for 1 second.{RESET}")
    try:
        # Record a small silent chunk to compute noise floor
        calib_data = sd.rec(int(0.8 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        calib_rms = np.sqrt(np.mean(calib_data**2))
        VAD_THRESHOLD = max(calib_rms * VAD_SENSITIVITY_MULTIPLIER, 0.0025)
        print(f"✅ {GREEN}Calibration complete!{RESET} Noise RMS: {calib_rms:.5f} | Threshold set to: {VAD_THRESHOLD:.5f}")
    except Exception as e:
        VAD_THRESHOLD = 0.003
        print(f"{YELLOW}⚠️  Calibration failed ({e}). Using default threshold: {VAD_THRESHOLD}{RESET}")

    # 3. Load model
    print(f"🧠 Loading Whisper model '{CYAN}{MODEL_SIZE}{RESET}' on CPU (Threads: {CPU_THREADS}, Compute: {COMPUTE_TYPE})...")
    
    try:
        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS
        )
        print(f"✅ Model loaded into RAM successfully!")
    except Exception as e:
        print(f"{RED}❌ Failed to load Whisper model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    # 4. Start background worker threads
    processing_thread = threading.Thread(target=audio_processing_worker, daemon=True)
    processing_thread.start()

    transcription_thread = threading.Thread(target=transcription_worker, args=(model,), daemon=True)
    transcription_thread.start()

    # 5. Set up Global Keyboard Hotkeys
    hotkeys_dict = {
        "<alt>+s": on_hotkey_pressed,
        "<cmd>+<shift>+s": on_hotkey_pressed
    }
    
    print(f"🎹 Registering background global hotkeys: {CYAN}Option+S{RESET} and {CYAN}Cmd+Shift+S{RESET}")
    
    try:
        listener = keyboard.GlobalHotKeys(hotkeys_dict)
        listener.start()
    except Exception as e:
        print(f"{RED}❌ Failed to register global keyboard listener: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    print(f"💡 {GREEN}Fallback:{RESET} You can also press {CYAN}Enter{RESET} in this terminal window to toggle recording.")
    print(f"\n[Ready] Sitting in background (Press {CYAN}Option+S{RESET}, {CYAN}Cmd+Shift+S{RESET}, or {CYAN}Enter{RESET} in terminal to toggle)...")
    
    # 6. Keep main thread alive and listen for Enter key fallback
    try:
        while True:
            input()
            on_hotkey_pressed()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{BLUE}Shutting down dictation utility...{RESET}")
        
        # Stop background workers
        raw_audio_queue.put(None)
        transcription_queue.put(None)
        
        processing_thread.join(timeout=2.0)
        transcription_thread.join(timeout=2.0)
        
        # Stop keyboard listener
        listener.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()
