import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
from huggingface_hub import InferenceClient

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

MODEL = "openai/whisper-small"  # use tiny if slow

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds


# -----------------------------
# RECORD AUDIO CHUNK
# -----------------------------
def record_chunk(duration=CHUNK_DURATION):
    print("🎤 Listening...")
    recording = sd.rec(int(duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='int16')
    sd.wait()
    return recording


# -----------------------------
# S2T USING HF
# -----------------------------
def speech_to_text(audio_array):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav.write(tmp.name, SAMPLE_RATE, audio_array)
        temp_path = tmp.name

    try:
        with open(temp_path, "rb") as f:
            result = client.automatic_speech_recognition(
                audio=f,
                model=MODEL
            )
        return result["text"]
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

            if text.strip():
                full_text += " " + text
                print("📝", full_text)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        print("\n📄 Final Transcript:\n", full_text)


# -----------------------------
if __name__ == "__main__":
    realtime_transcription()