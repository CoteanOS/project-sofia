#!/usr/bin/env python3
# voice.py - "Hey Sofia" hands-free loop.
# wake (openWakeWord) -> record -> whisper -> server.py -> Kokoro TTS.
# Fully local. Saves "remember/call me" statements to memory and preloads them.
import os, re, subprocess
import numpy as np
import requests
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
from kokoro import KPipeline
import soundfile as sf
from memory import ingest, retrieve

MODEL_PATH = os.path.expanduser("~/code/sofia/hey_sofia.onnx")
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
kokoro = KPipeline(lang_code='a')   # 'a' = American English (voice af_heart)
stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK)

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF"
                    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF"
                    "\U00002B00-\U00002BFF\uFE0F]+")

def read_chunk():
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
    audio = None
    for _, _, a in kokoro(text, voice="af_heart"):
        audio = a
    if audio is None:
        return
    sf.write("/tmp/sofia.wav", audio, 24000)
    subprocess.run(["afplay", "/tmp/sofia.wav"])

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
    first = True
    while True:
        audio, heard = record_turn()
        if not heard:
            return
        text = transcribe(audio)
        if not text:
            if first:
                speak("I didn't catch that.")
                first = False
                continue
            return
        print(f"  you: {text}")
        if text.lower().strip(" .!?") in ("stop", "never mind", "cancel", "goodbye", "bye"):
            return
        saved = save_if_memory(text)
        if saved:
            print(f"  [memory saved] {saved}")
            speak("Got it, I'll remember that.")
            first = False
            continue
        history.append({"role": "user", "content": text})
        try:
            reply = ask_sofia(history)
        except Exception as e:
            speak("Sofia isn't reachable. Is the server running?")
            print(f"  error: {e}")
            history.pop()
            return
        history.append({"role": "assistant", "content": reply})
        del history[:-16]
        print(f"  sofia: {reply[:200]}")
        speak(reply)
        first = False

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
