#!/usr/bin/env python3
"""
dictate.py — Blazing-Fast On-Demand Voice-to-Text Dictation

A production-grade, real-time dictation utility optimized for macOS on Intel x86_64.
Keeps a Whisper model persistently in RAM for instant inference.
Uses adaptive VAD with pre-speech buffering for accurate word-onset detection.
Streams provisional transcription with diff-based typing for seamless real-time output.

Usage:
    .venv/bin/python dictate.py

    Press Option+S or Cmd+Shift+S (or Enter in terminal) to toggle dictation.
    Click into any text field before speaking — text is typed at the active cursor.
"""

from __future__ import annotations

import sys
import time
import queue
import threading
import traceback
from collections import deque

import numpy as np
import sounddevice as sd
from pynput import keyboard
from faster_whisper import WhisperModel


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Tune these for your hardware and preferences
# ═══════════════════════════════════════════════════════════════════════════════

# ── Audio Capture ─────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000       # Hz — Whisper expects 16 kHz
CHANNELS: int = 1               # Mono
BLOCKSIZE: int = 1024           # Samples per chunk (64 ms at 16 kHz)

# ── Whisper Model ─────────────────────────────────────────────────────────────
# Model accuracy / speed ladder (English-only):
#   "tiny.en"   → Fastest,  lowest accuracy  (~39 MB)
#   "base.en"   → Fast,     decent accuracy   (~74 MB)
#   "small.en"  → Balanced, good accuracy     (~461 MB)  ← RECOMMENDED
#   "medium.en" → Slower,   high accuracy     (~1.5 GB)
MODEL_SIZE: str = "small.en"
DEVICE: str = "cpu"
COMPUTE_TYPE: str = "int8"      # int8 is 2-3× faster than float32 on Intel x86_64
CPU_THREADS: int = 6

# ── Inference Tuning ──────────────────────────────────────────────────────────
BEAM_SIZE_STREAMING: int = 1    # Greedy decoding for streaming previews (fastest)
BEAM_SIZE_FINAL: int = 5        # Beam search for final commits (most accurate)

# Prime Whisper with correct spellings of names / terms you use often.
# This dramatically improves recognition of proper nouns.
INITIAL_PROMPT: str = (
    "Hello, my name is Muhammad. I am from Sargodha. "
    "I am dictating clear, well-punctuated English text."
)

# ── Voice Activity Detection ─────────────────────────────────────────────────
VAD_SENSITIVITY: float = 1.8    # Noise-floor multiplier (lower = more sensitive; 1.3–2.5)
AUTO_COMMIT_DELAY: float = 0.5  # Seconds of silence before finalizing a segment
STREAMING_INTERVAL: float = 0.4 # Seconds between streaming transcription updates
PRE_SPEECH_CHUNKS: int = 5      # Chunks retained before speech onset (~320 ms cushion)
NEWLINE_PAUSE: float = 3.0      # Seconds of pause before inserting a newline


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Terminal ANSI colours
_G = "\033[92m"     # green
_R = "\033[91m"     # red
_Y = "\033[93m"     # yellow
_B = "\033[94m"     # blue
_C = "\033[96m"     # cyan
_BD = "\033[1m"     # bold
_RS = "\033[0m"     # reset

# Sentinel objects for the raw-audio queue
_FLUSH = object()   # flush remaining speech buffer and commit
_STOP = object()    # shut down the processing worker

# State machine
_S_IDLE = "IDLE"
_S_REC = "RECORDING"
_S_XSCR = "TRANSCRIBING"


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

_state: str = _S_IDLE
_state_lock = threading.Lock()

_raw_q: queue.Queue = queue.Queue()
_xscr_q: queue.Queue = queue.Queue()
_stream: sd.InputStream | None = None

_vad_threshold: float = 0.003
_latest_text: str = ""

_kb = keyboard.Controller()

# Duration of one audio chunk in seconds (computed once)
_CHUNK_SEC: float = BLOCKSIZE / SAMPLE_RATE


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _audio_cb(indata: np.ndarray, frames: int, time_info, status) -> None:
    """PortAudio callback — pushes raw audio chunks to the processing queue."""
    if status:
        print(f"\n{_Y}⚠️  Audio: {status}{_RS}", file=sys.stderr)
    _raw_q.put(indata.copy())


def _enqueue_streaming(audio: np.ndarray, prepend_nl: bool) -> None:
    """
    Queue a streaming (provisional) transcription request.
    Drops stale streaming items to prevent backlog while preserving
    final commits and shutdown signals.
    """
    kept: list = []
    while not _xscr_q.empty():
        try:
            item = _xscr_q.get_nowait()
            if item is None or item[1]:     # shutdown signal or is_final
                kept.append(item)
        except queue.Empty:
            break
    for item in kept:
        _xscr_q.put(item)
    _xscr_q.put((audio, False, prepend_nl))


def _vad_worker() -> None:
    """
    VAD worker thread.

    Reads raw audio chunks, detects speech using calibrated RMS thresholds,
    maintains a pre-speech ring buffer for clean word onsets, and dispatches
    audio segments for transcription.
    """
    global _latest_text

    speech_buf: list[np.ndarray] = []
    pre_speech: deque[np.ndarray] = deque(maxlen=PRE_SPEECH_CHUNKS)
    is_speaking = False
    silence_s = 0.0
    last_stream_t = 0.0
    last_commit_t = 0.0
    prepend_nl = False

    # Throttle terminal redraws to ≤ 10 Hz
    last_print_t = 0.0
    prev_speaking = False

    while True:
        try:
            chunk = _raw_q.get()

            # ── Shutdown ──
            if chunk is _STOP:
                _raw_q.task_done()
                break

            # ── Flush (recording stopped) ──
            if chunk is _FLUSH:
                if speech_buf:
                    audio = np.concatenate(speech_buf).flatten()
                    if np.sqrt(np.mean(audio ** 2)) >= 1e-4:
                        _xscr_q.put((audio, True, prepend_nl))
                    speech_buf.clear()
                is_speaking = False
                silence_s = 0.0
                pre_speech.clear()
                _raw_q.task_done()
                continue

            # ── Normal audio chunk ──
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > _vad_threshold:
                # Speech detected
                if not is_speaking:
                    is_speaking = True
                    speech_buf = list(pre_speech)   # prepend onset cushion
                    pre_speech.clear()
                    last_stream_t = time.monotonic()
                    prepend_nl = (
                        last_commit_t > 0
                        and time.monotonic() - last_commit_t >= NEWLINE_PAUSE
                    )

                speech_buf.append(chunk)
                silence_s = 0.0

                # Periodic streaming transcription
                now = time.monotonic()
                if now - last_stream_t >= STREAMING_INTERVAL:
                    _enqueue_streaming(
                        np.concatenate(speech_buf).flatten(), prepend_nl,
                    )
                    last_stream_t = now
            else:
                # Silence
                if is_speaking:
                    speech_buf.append(chunk)
                    silence_s += _CHUNK_SEC
                    if silence_s >= AUTO_COMMIT_DELAY:
                        _xscr_q.put((
                            np.concatenate(speech_buf).flatten(),
                            True,
                            prepend_nl,
                        ))
                        last_commit_t = time.monotonic()
                        speech_buf.clear()
                        is_speaking = False
                        silence_s = 0.0
                        pre_speech.clear()
                else:
                    pre_speech.append(chunk)     # deque auto-evicts oldest

            # ── Status bar (throttled) ──
            now = time.monotonic()
            if (now - last_print_t >= 0.1) or (is_speaking != prev_speaking):
                buf_s = len(speech_buf) * _CHUNK_SEC
                icon = f"{_G}🟢{_RS}" if is_speaking else f"{_Y}⚪{_RS}"
                txt = _latest_text
                if len(txt) > 20:
                    txt = "…" + txt[-19:]
                print(
                    f"\rVAD: {icon} | RMS: {rms:.4f}/{_vad_threshold:.4f}"
                    f" | Buf: {buf_s:.1f}s | \"{_C}{txt}{_RS}\"    ",
                    end="", flush=True,
                )
                last_print_t = now
                prev_speaking = is_speaking

            _raw_q.task_done()

        except Exception as exc:
            print(f"\n{_R}❌ VAD error: {exc}{_RS}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
#  TYPING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _diff_type(old: str, new: str) -> None:
    """
    Minimal-edit typing: finds the longest common prefix between *old* and
    *new*, backspaces the divergent suffix of *old*, then types the new suffix.
    """
    if old == new:
        return

    # Common prefix length
    pfx = 0
    for a, b in zip(old, new):
        if a != b:
            break
        pfx += 1

    # Backspace removed characters
    to_del = len(old) - pfx
    if to_del:
        for _ in range(to_del):
            _kb.press(keyboard.Key.backspace)
            _kb.release(keyboard.Key.backspace)
            time.sleep(0.003)           # 3 ms — prevents macOS event coalescing

    # Type new characters
    to_type = new[pfx:]
    if to_type:
        _kb.type(to_type)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _xscr_worker(model: WhisperModel) -> None:
    """
    Background thread that runs Whisper inference on queued audio segments.

    Uses **greedy decoding** for streaming previews (lowest latency) and
    **beam search** for final commits (highest accuracy).
    """
    global _state, _latest_text

    typed = ""

    while True:
        item = _xscr_q.get()
        if item is None:
            _xscr_q.task_done()
            break

        audio_data, is_final, do_newline = item

        try:
            beam = BEAM_SIZE_FINAL if is_final else BEAM_SIZE_STREAMING
            segments, _ = model.transcribe(
                audio_data,
                beam_size=beam,
                language="en",
                initial_prompt=INITIAL_PROMPT,
                without_timestamps=True,
            )
            full = "".join(seg.text for seg in segments).strip()

            if full:
                if do_newline and not typed:
                    _kb.type("\n")

                _diff_type(typed, full)

                if is_final:
                    _kb.type(" ")
                    typed = ""
                    _latest_text = ""
                    print(f"\n✨ {_G}{_BD}[Committed]{_RS} \"{full}\"")
                else:
                    typed = full
                    _latest_text = full
            elif is_final:
                # Empty final — clear any provisional typing
                _diff_type(typed, "")
                typed = ""
                _latest_text = ""

        except Exception as exc:
            print(f"\n{_R}❌ Transcription error: {exc}{_RS}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            _xscr_q.task_done()
            with _state_lock:
                if (
                    _state == _S_XSCR
                    and _xscr_q.empty()
                    and _stream is None
                ):
                    _state = _S_IDLE
                    print(
                        f"\n{_G}[Ready]{_RS} Press {_C}Option+S{_RS}, "
                        f"{_C}Cmd+Shift+S{_RS}, or {_C}Enter{_RS} to toggle…"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RECORDING CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

def _toggle() -> None:
    """Thread-safe state machine: IDLE → RECORDING → TRANSCRIBING."""
    global _state, _stream

    with _state_lock:
        if _state == _S_IDLE:
            # Drain stale audio
            while not _raw_q.empty():
                try:
                    _raw_q.get_nowait()
                except queue.Empty:
                    break

            try:
                _stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=BLOCKSIZE,
                    callback=_audio_cb,
                )
                _stream.start()
                _state = _S_REC
                print(
                    f"\n🟢 {_G}{_BD}Listening…{_RS}"
                    f" (Press hotkey or Enter to stop)"
                )
            except Exception as exc:
                print(f"\n{_R}❌ Mic error: {exc}{_RS}", file=sys.stderr)
                print(
                    f"{_Y}💡 Grant Microphone access in "
                    f"System Settings → Privacy & Security → Microphone{_RS}",
                    file=sys.stderr,
                )

        elif _state == _S_REC:
            _state = _S_XSCR
            print(f"\n⏹️  {_B}Stopping…{_RS}")
            if _stream:
                try:
                    _stream.stop()
                    _stream.close()
                except Exception as exc:
                    print(f"{_R}❌ Stream error: {exc}{_RS}", file=sys.stderr)
                _stream = None
            _raw_q.put(_FLUSH)

        elif _state == _S_XSCR:
            print(
                f"\n{_Y}⏳ Finalizing, please wait…{_RS}",
                end="", flush=True,
            )


def _on_hotkey() -> None:
    """Hotkey callback — dispatches toggle to a daemon thread."""
    threading.Thread(target=_toggle, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _vad_threshold

    print(f"\n{_B}{_BD}═══ Whisper Dictation ═══{_RS}")
    print(
        f"{_C}Model: {MODEL_SIZE} | Compute: {COMPUTE_TYPE} | "
        f"Threads: {CPU_THREADS}{_RS}\n"
    )

    # ── Audio device ──────────────────────────────────────────────────────────
    try:
        dev = sd.query_devices(kind="input")
        print(f"🎤 Input: {_C}{dev.get('name', 'Default')}{_RS}")
    except Exception as exc:
        print(f"{_R}❌ No audio input: {exc}{_RS}", file=sys.stderr)
        sys.exit(1)

    # ── Noise calibration ─────────────────────────────────────────────────────
    print("🎙️  Calibrating noise floor (stay silent)…")
    try:
        calib = sd.rec(
            int(0.8 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()
        noise_rms = float(np.sqrt(np.mean(calib ** 2)))
        _vad_threshold = max(noise_rms * VAD_SENSITIVITY, 0.0025)
        print(f"✅ Noise: {noise_rms:.5f} → Threshold: {_vad_threshold:.5f}")
    except Exception as exc:
        _vad_threshold = 0.003
        print(f"{_Y}⚠️  Calibration failed: {exc} (using {_vad_threshold}){_RS}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"🧠 Loading {_C}{MODEL_SIZE}{_RS} ({COMPUTE_TYPE})…")
    model: WhisperModel | None = None
    try:
        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
        )
        print(f"✅ Model loaded and cached in RAM")
    except Exception as exc:
        if COMPUTE_TYPE != "float32":
            print(
                f"{_Y}⚠️  {COMPUTE_TYPE} failed ({exc}), "
                f"falling back to float32…{_RS}"
            )
            try:
                model = WhisperModel(
                    MODEL_SIZE,
                    device=DEVICE,
                    compute_type="float32",
                    cpu_threads=CPU_THREADS,
                )
                print(f"✅ Model loaded (float32 fallback)")
            except Exception as exc2:
                print(
                    f"{_R}❌ Model load failed: {exc2}{_RS}", file=sys.stderr
                )
                sys.exit(1)
        else:
            print(f"{_R}❌ Model load failed: {exc}{_RS}", file=sys.stderr)
            sys.exit(1)

    assert model is not None

    # ── Start workers ─────────────────────────────────────────────────────────
    threading.Thread(target=_vad_worker, daemon=True).start()
    threading.Thread(target=_xscr_worker, args=(model,), daemon=True).start()

    # ── Global hotkeys ────────────────────────────────────────────────────────
    hotkeys = {
        "<alt>+s": _on_hotkey,
        "<cmd>+<shift>+s": _on_hotkey,
    }
    print(f"🎹 Hotkeys: {_C}Option+S{_RS} / {_C}Cmd+Shift+S{_RS}")
    try:
        listener = keyboard.GlobalHotKeys(hotkeys)
        listener.start()
    except Exception as exc:
        print(f"{_R}❌ Hotkey error: {exc}{_RS}", file=sys.stderr)
        sys.exit(1)

    print(f"💡 You can also press {_C}Enter{_RS} in this terminal to toggle.")
    print(
        f"\n{_G}[Ready]{_RS} Click into a text field, "
        f"then press your hotkey to start dictating.\n"
    )

    # ── Main loop (Enter fallback) ────────────────────────────────────────────
    try:
        while True:
            input()
            _on_hotkey()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{_B}Shutting down…{_RS}")
        _raw_q.put(_STOP)
        _xscr_q.put(None)
        listener.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()
