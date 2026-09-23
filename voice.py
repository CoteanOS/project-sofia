#!/usr/bin/env python3
# voice.py - "Hey Sofia" hands-free loop.
# wake (openWakeWord) -> record -> whisper -> server.py -> Qwen3-TTS (MLX).
# Fully local. Saves "remember/call me" statements to memory and preloads them.
import os, re, subprocess, time
import numpy as np
import requests
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
from mlx_audio.tts.utils import load_model
import soundfile as sf
from memory import ingest, retrieve

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hey_sofia.onnx")
KEYWORD    = "hey_sofia"
THRESHOLD  = 0.85               # raise if it false-triggers, lower if it misses you
SOFIA_URL  = "http://127.0.0.1:8000/v1/chat/completions"

RATE           = 16000
CHUNK          = 1280           # openWakeWord wants 1280-sample (80ms) int16 chunks
SILENCE_RMS    = 600            # below this = silence
SILENCE_CHUNKS = 12             # ~1s of quiet ends your turn
MAX_CHUNKS     = 150            # ~12s hard cap per turn
COOLDOWN       = 8              # chunks ignored right after a conversation

print("Loading models...")
openwakeword.utils.download_models()
oww = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
stt = WhisperModel("base.en", device="cpu", compute_type="int8")
# Qwen3-TTS via MLX-Audio: Metal-accelerated on Apple Silicon, so it's fast AND expressive.
TTS_MODEL = "mlx-community/Soprano-1.1-80M-bf16"   # tiny 80M, near-instant
VOICE_STYLE = "a slow, somber British woman, low and measured, weary and subdued"
TEMPERATURE = 0.2   # lower = flatter, more monotone/measured delivery
SPEED       = 0.85  # <1 = slower playback
tts = load_model(TTS_MODEL)
stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK)
_muted = False

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF"
                    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF"
                    "\U00002B00-\U00002BFF\uFE0F]+")

import time as _time

def _mic_stop():
    try:
        if stream.active:
            stream.stop()
    except Exception:
        pass

def _mic_start():
    # CoreAudio can transiently refuse a stream (-9986). Retry a few times.
    for _ in range(5):
        try:
            if not stream.active:
                stream.start()
            return
        except Exception:
            _time.sleep(0.2)
    # last try, let it raise only if it still fails
    if not stream.active:
        stream.start()

def read_chunk():
    if _muted:
        return np.zeros(CHUNK, dtype="int16")
    data, _ = stream.read(CHUNK)
    return data.flatten()

def rms(frame):
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

def record_turn():
    frames, silent, speech_chunks = [], 0, 0
    for _ in range(MAX_CHUNKS):
        chunk = read_chunk()
        frames.append(chunk)
        if rms(chunk) > SILENCE_RMS:
            speech_chunks += 1
            silent = 0
        elif speech_chunks >= 3:
            silent += 1
            if silent > SILENCE_CHUNKS:
                break
    heard = speech_chunks >= 3
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    return audio, heard

def transcribe(audio):
    segments, _ = stt.transcribe(audio, language="en")
    return " ".join(s.text for s in segments).strip()

def ask_sofia(history):
    r = requests.post(SOFIA_URL, json={"model": "sofia", "messages": history}, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def strip_think(t):
    # kill qwen reasoning leaks: <think>..</think> blocks and bare /think //no_think switches
    t = re.sub(r"<think>[\\s\\S]*?</think>", " ", t, flags=re.I)
    t = re.sub(r"/no_think|/think", " ", t, flags=re.I)
    return re.sub(r"\\s+", " ", t).strip()

def clean_for_speech(t):
    t = _EMOJI.sub("", t)
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"[*_#>~|]", "", t)
    t = re.sub(r"^\s*[-+]\s+", "", t, flags=re.M)
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()

def speak(text):
    text = clean_for_speech(text)
    if not text:
        return
    global _muted
    _muted = True                     # mute her whole busy window: generate + play
    try:
        gen_iter = tts.generate(text=text)
        segments, sr = [], 24000
        for result in gen_iter:
            segments.append(np.asarray(result.audio, dtype=np.float32))
            sr = int(getattr(result, "sample_rate", None) or getattr(result, "sr", sr))
        if not segments:
            return
        audio = np.concatenate(segments).astype(np.float32)
        sf.write("/tmp/sofia.wav", audio, sr, subtype="PCM_16")
        subprocess.run(["afplay", "/tmp/sofia.wav"], check=False)
        time.sleep(0.35)              # let the speaker/echo tail die
    finally:
        try:
            for _ in range(50):
                avail = stream.read_available
                if not avail:
                    break
                stream.read(avail)
        except Exception:
            pass
        _muted = False                # ALWAYS reset, even on early return/error

def save_if_memory(text):
    low = text.lower().strip()
    for trigger in ("remember that ", "remember ", "call me ", "my name is ", "note that "):
        if low.startswith(trigger):
            ingest(text.strip(), source="user")
            return text.strip()
    return None

def conversation(history):
    speak("Yes?")
    facts = retrieve("how to address the user, the user's name and preferences")
    if facts and not any(m.get("role") == "system" for m in history):
        history.insert(0, {"role": "system",
            "content": "Known facts about the user (honor these):\n" + "\n".join(facts)})
    misses = 0
    while True:
        audio, heard = record_turn()
        if not heard:
            # A pause is not a goodbye. Wait through a couple of quiet windows
            # before giving up, so she does not drop you the moment you think.
            misses += 1
            if misses >= 3:
                return
            continue
        misses = 0
        text = transcribe(audio)
        if not text:
            if misses < 3:
                misses += 1
                continue
            return
        print(f"  you: {text}")
        if text.lower().strip(" .!?") in ("stop", "never mind", "cancel", "goodbye", "bye"):
            return
        saved = save_if_memory(text)
        if saved:
            print(f"  [memory saved] {saved}")
            speak("Got it, I'll remember that.")
            continue
        history.append({"role": "user", "content": text})
        try:
            reply = ask_sofia(history)
        except Exception as e:
            speak("Sofia isn't reachable. Is the server running?")
            print(f"  error: {e}")
            history.pop()
            return
        reply = strip_think(reply)
        history.append({"role": "assistant", "content": reply})
        del history[:-16]
        print(f"  sofia: {reply[:200]}")
        speak(reply)

def main():
    stream.start()
    print('Listening for "Hey Sofia"...  (Ctrl+C to quit)')
    cooldown, hits, history = 0, 0, []
    try:
        while True:
            chunk = read_chunk()
            if cooldown > 0:
                cooldown -= 1
                continue
            score = oww.predict(chunk).get(KEYWORD, 0.0)
            hits = hits + 1 if score > THRESHOLD else 0
            if hits >= 2:
                hits = 0
                print(f"  [wake {score:.2f}]")
                conversation(history)
                cooldown = COOLDOWN
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()

if __name__ == "__main__":
    main()
