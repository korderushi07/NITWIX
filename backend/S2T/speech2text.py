import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
from faster_whisper import WhisperModel

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds

# 🔥 Load local Whisper model
# Use "tiny" for fastest, "small" for better accuracy
model = WhisperModel("tiny", compute_type="int8")


# -----------------------------
# RECORD AUDIO CHUNK
# -----------------------------
def record_chunk(duration=CHUNK_DURATION):
    print("🎤 Listening...")
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    return recording


# -----------------------------
# SPEECH TO TEXT (LOCAL)
# -----------------------------
def speech_to_text(audio_array):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav.write(tmp.name, SAMPLE_RATE, audio_array)
        temp_path = tmp.name

    try:
        segments, _ = model.transcribe(temp_path)

        text = ""
        for segment in segments:
            text += segment.text

        return text.strip()

    finally:
        os.remove(temp_path)


# -----------------------------
# REAL-TIME LOOP
# -----------------------------
def realtime_transcription():
    full_text = ""

    print("🚀 Real-time transcription started (Ctrl+C to stop)\n")

    try:
        while True:
            audio_chunk = record_chunk()

            text = speech_to_text(audio_chunk)

            if text:
                full_text += " " + text
                print("📝", full_text)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        print("\n📄 Final Transcript:\n", full_text)

    return full_text


 