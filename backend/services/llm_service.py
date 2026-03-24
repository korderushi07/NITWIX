from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API")
client = Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are a professional court assistant helping a clerk.
- Convert raw speech into structured, clear, formal responses.
- Be precise and legally appropriate.
- Remove filler words.
"""

def call_llm(transcript: str):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # 🔥 best on Groq
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}"