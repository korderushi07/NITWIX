import sounddevice as sd

def record_chunk(duration=CHUNK_DURATION):
    print("🎤 Listening...")
    recording = sd.rec(int(duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='int16')
    sd.wait()
    return recording