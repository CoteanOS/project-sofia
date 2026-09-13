import re
#!/usr/bin/env python3
# voice.py - "Hey Sofia" hands-free loop (openWakeWord, fully local, no account).
# wake -> record until silence -> faster-whisper -> server.py -> macOS `say`.
import os, subprocess
import numpy as np
import requests
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel


from kokoro import KPipeline
import soundfile as _sf
_kokoro = KPipeline(lang_code='a')   # British English, loaded once
MODEL_PATH = os.path.expanduser("~/code/sofia/hey_sofia.onnx")
KEYWORD    = "hey_sofia"          # = the .onnx filename stem
THRESHOLD  = 0.8                  # raise toward 0.7 if it false-triggers
SOFIA_URL  = "http://127.0.0.1:8000/v1/chat/completions"

RATE           = 16000
CHUNK          = 1280            # openWakeWord wants 1280-sample (80ms) int16 chunks
SILENCE_RMS    = 600             # below this = silence
SILENCE_CHUNKS = 12              # ~1s of quiet ends your turn
MAX_CHUNKS     = 150             # ~12s hard cap per turn
COOLDOWN       = 8              # chunks to ignore right after a conversation

print("Loading models...")
openwakeword.utils.download_models()                       # shared feature models, one-time
oww = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
stt = WhisperModel("base.en", device="cpu", compute_type="int8")

stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK)

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
    heard = speech_chunks >= 3          # need real speech, not one noise spike
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    return audio, heard

def transcribe(audio):
    segments, _ = stt.transcribe(audio, language="en")
    return " ".join(s.text for s in segments).strip()

def ask_sofia(text):
    r = requests.post(SOFIA_URL, json={
        "model": "sofia",
        "messages": [{"role": "user", "content": text}],
    }, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

import re as _re
_EMOJI = _re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002B00-\U00002BFF\uFE0F]+")
def strip_emoji(t):
    return _EMOJI.sub("", t).strip()

def clean_for_speech(t):
    t = strip_emoji(t)
    t = re.sub(r"```[\s\S]*?```", " ", t)      # drop code blocks entirely
    t = re.sub(r"`([^`]*)`", r"\1", t)          # inline code -> its text
    t = re.sub(r"[*_#>~|]", "", t)               # markdown symbols
    t = re.sub(r"^\s*[-+]\s+", "", t, flags=re.M)  # bullet dashes
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)    # links -> link text
    t = re.sub(r"\s+", " ", t)                   # collapse whitespace
    return t.strip()

def speak(text):
    text = clean_for_speech(text)
    if not text:
        return
    audio = None
    for _, _, a in _kokoro(text, voice="af_heart"):
        audio = a
    if audio is None:
        return
    _sf.write("/tmp/sofia.wav", audio, 24000)
    subprocess.run(["afplay", "/tmp/sofia.wav"])

def conversation():
    speak("Yes?")
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
        try:
            reply = ask_sofia(text)
        except Exception as e:
            speak("Sofia isn't reachable. Is the server running?")
            print(f"  error: {e}")
            return
        print(f"  sofia: {reply[:200]}")
        speak(reply)
        first = False

def main():
    stream.start()
    print('Listening for "Hey Sofia"...  (Ctrl+C to quit)')
    cooldown = 0
    try:
        while True:
            chunk = read_chunk()
            if cooldown > 0:
                cooldown -= 1
                continue
            score = oww.predict(chunk).get(KEYWORD, 0.0)
            if score > THRESHOLD:
                print(f"  [wake {score:.2f}]")
                conversation()
                cooldown = COOLDOWN
                print('Listening for "Hey Sofia"...')
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()

if __name__ == "__main__":
    main()
