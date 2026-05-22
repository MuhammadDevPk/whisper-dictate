#!/usr/bin/env python3
"""
dictate.py - On-Demand Continuous Voice-to-Text Dictation Utility

Requirements:
1. Persistent accelerated base.en Whisper model loaded strictly on CPU (6 threads, float32).
2. Global Keyboard Hook (Option + S or Cmd + Shift + S) or Enter to toggle continuous dictation.
3. Sound capture buffer via sounddevice InputStream at 16kHz, mono channel.
4. Voice Activity Detection (VAD) via rolling-minimum RMS tracking to prevent state lockups.
5. Real-time streaming transcription running every 1.0s with diff-based typing and backspace correction.
6. Auto-injection of transcribed text at active cursor location via pynput.keyboard.Controller.
"""

import sys
import time
import queue
import threading
import traceback
import numpy as np
import sounddevice as sd
from pynput import keyboard
from collections import deque
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
MODEL_SIZE = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
CPU_THREADS = 6

# VAD Settings
AUTO_COMMIT_DELAY = 0.8  # Commit audio segment after 0.8 seconds of silence
STREAMING_INTERVAL = 1.0  # Transcribe provisional text every 1.0s of continuous speech

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

# pynput Keyboard Controller
keyboard_controller = keyboard.Controller()


def audio_callback(indata, frames, time_info, status):
    """Callback function for sounddevice to capture audio chunks."""
    if status:
        print(f"\n{YELLOW}⚠️  Audio stream warning: {status}{RESET}", file=sys.stderr)
    raw_audio_queue.put((indata.copy(), False))


def push_streaming_audio(audio_data):
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
        
    # Queue the new provisional streaming audio (is_final = False)
    transcription_queue.put((audio_data, False))


def audio_processing_worker():
    """
    Dedicated worker thread that processes raw audio chunks from raw_audio_queue.
    Performs rolling-minimum RMS-based VAD filtering and groups active speech frames.
    Commits completed segments and updates provisional streaming chunks.
    """
    speech_buffer = []
    has_speech = False
    silence_duration = 0.0
    last_stream_time = 0.0
    
    # 60 chunks of 30ms is ~1.8 seconds of history
    rms_history = deque(maxlen=60)
    
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
                        transcription_queue.put((audio_data, True))
                    speech_buffer = []
                has_speech = False
                silence_duration = 0.0
                rms_history.clear()
                raw_audio_queue.task_done()
                continue
                
            # 2. Process Standard Audio Chunk
            rms = np.sqrt(np.mean(chunk**2))
            rms_history.append(rms)
            
            # Noise floor is the minimum volume level seen in the last 1.8 seconds
            noise_floor = min(rms_history)
            
            # Speaking threshold is calibrated dynamically
            threshold = max(noise_floor * 1.8, 0.002)
            chunk_duration = len(chunk) / SAMPLE_RATE
            
            if rms > threshold:
                speech_buffer.append(chunk)
                silence_duration = 0.0
                
                # Speech starts
                if not has_speech:
                    has_speech = True
                    last_stream_time = time.time()
                else:
                    # Periodically stream the accumulated buffer to Whisper for real-time feedback
                    current_time = time.time()
                    if current_time - last_stream_time >= STREAMING_INTERVAL:
                        audio_data = np.concatenate(speech_buffer, axis=0).flatten()
                        push_streaming_audio(audio_data)
                        last_stream_time = current_time
            else:
                if has_speech:
                    speech_buffer.append(chunk)
                    silence_duration += chunk_duration
                    
                    # If silence duration exceeds the commit delay, finalize the segment
                    if silence_duration >= AUTO_COMMIT_DELAY:
                        audio_data = np.concatenate(speech_buffer, axis=0).flatten()
                        transcription_queue.put((audio_data, True))
                        
                        # Reset for next speech segment
                        speech_buffer = []
                        has_speech = False
                        silence_duration = 0.0
                else:
                    # Discard quiet chunks during long silence
                    pass
            
            # Visual real-time status bar
            status_text = f"{GREEN}🟢 SPEAKING{RESET}" if has_speech else f"{YELLOW}⚪ SILENT{RESET}"
            print(f"\r📊 Noise Floor: {noise_floor:.5f} | Threshold: {threshold:.5f} | RMS: {rms:.5f} | Buffer: {len(speech_buffer)} chunks | {status_text}", end="", flush=True)
            
            raw_audio_queue.task_done()
            
        except Exception as e:
            print(f"\n{RED}❌ Error in audio processing thread: {e}{RESET}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def update_typing(old_text, new_text):
    """
    Dynamically typing diffs: compares old_text and new_text,
    sends Backspaces for modified letters, and types new letters.
    """
    # Find common prefix length
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
    global current_state
    typed_text = ""
    
    while True:
        item = transcription_queue.get()
        if item is None:
            transcription_queue.task_done()
            break
            
        audio_data, is_final = item
        
        try:
            # Print log entry
            label = "Final" if is_final else "Stream"
            print(f"\n🧠 {BOLD}Transcribing ({label})...{RESET}", end="", flush=True)
            
            segments, info = model.transcribe(
                audio_data,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            text_segments = [seg.text for seg in segments]
            full_text = "".join(text_segments).strip()
            
            # Print transcription results
            print(f"\r🧠 {BOLD}Transcribing ({label}): Done.{RESET}")
            if full_text:
                print(f"✨ {GREEN if is_final else CYAN}{BOLD}[{label}]{RESET} \"{full_text}\"")
                
                # Perform the differential typing
                update_typing(typed_text, full_text)
                
                if is_final:
                    # Append a trailing space for next segment and clear state
                    keyboard_controller.type(" ")
                    typed_text = ""
                else:
                    typed_text = full_text
            else:
                if is_final:
                    # If empty final text, make sure we clear any provisional typing
                    update_typing(typed_text, "")
                    typed_text = ""
                    print(f"⚠️  {YELLOW}No speech committed.{RESET}")
                else:
                    # Don't erase yet if it's just a brief quiet stream segment
                    pass
                    
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
    print(f"{BLUE}{BOLD}=== Whisper On-Demand Dictation Utility ==={RESET}")
    
    # 1. Query audio input device
    try:
        input_device = sd.query_devices(kind="input")
        device_name = input_device.get("name", "Default Device")
        print(f"🎤 Default Input Device: {CYAN}{device_name}{RESET}")
    except Exception as e:
        print(f"{RED}❌ No input audio devices found or sounddevice error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    # 2. Load model
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

    # 3. Start background worker threads
    processing_thread = threading.Thread(target=audio_processing_worker, daemon=True)
    processing_thread.start()

    transcription_thread = threading.Thread(target=transcription_worker, args=(model,), daemon=True)
    transcription_thread.start()

    # 4. Set up Global Keyboard Hotkeys
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
    
    # 5. Keep main thread alive and listen for Enter key fallback
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
