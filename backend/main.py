from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os

from faster_whisper import WhisperModel
from services.agent_team import generate_response

 
 
 

from S2T.speech2text import speech_to_text
 

app = FastAPI()

# 🔐 Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = WhisperModel("tiny", compute_type="int8")

def speech_to_text(audio_path):
    segments, _ = model.transcribe(audio_path)
    return " ".join([seg.text for seg in segments]).strip()


@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        tmp.write(await file.read())

    try:
        transcript = speech_to_text(temp_path)
        response = generate_response(transcript)

        return {
            "transcript": transcript,
            "ai_response": response
        }

    finally:
        os.remove(temp_path)




def save_temp_file(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(upload_file.file.read())
        return tmp.name


@app.post("/process-multi-audio/")
async def process_multi_audio(
    speaker1: UploadFile = File(...),
    speaker2: UploadFile = File(...),
    speaker3: UploadFile = File(...)
):
    paths = []

    try:
        # 🔥 Save files
        path1 = save_temp_file(speaker1)
        path2 = save_temp_file(speaker2)
        path3 = save_temp_file(speaker3)

        paths.extend([path1, path2, path3])

        # 🔥 STT
        text1 = speech_to_text(path1)
        text2 = speech_to_text(path2)
        text3 = speech_to_text(path3)

        # 🔥 AI Processing (per speaker)
        res1 = generate_response(text1)
        res2 = generate_response(text2)
        res3 = generate_response(text3)

        # 🔥 Final Combined Output
        combined_prompt = f"""
        There are three speakers in a courtroom:

        Speaker 1: {res1}
        Speaker 2: {res2}
        Speaker 3: {res3}

        Generate a structured courtroom summary:
        - Who said what
        - Key arguments
        - Give the final under standing as Speaker 1, Speaker 2, Speaker 3
        """

        final_output = generate_response(combined_prompt)

        return {
            "speaker_1": {"transcript": text1, "ai": res1},
            "speaker_2": {"transcript": text2, "ai": res2},
            "speaker_3": {"transcript": text3, "ai": res3},
            "final_summary": final_output
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        for path in paths:
            if os.path.exists(path):
                os.remove(path)