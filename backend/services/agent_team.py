from services.llm_service import call_llm

# Step 1: Clean transcript
def clean_text(text: str):
    prompt = f"""
    Clean the following transcript:
    - Remove filler words (uh, um, noise)
    - Fix grammar
    - Keep meaning same

    TEXT:
    {text}
    """
    return call_llm(prompt)


# Step 2: Convert to legal format
def format_text(text: str):
    prompt = f"""
    Convert into formal court language:
    - Structured
    - Professional
    - Legally appropriate

    TEXT:
    {text}
    """
    return call_llm(prompt)


# 🔥 Pipeline (Agent Team Replacement)
def generate_response(transcript: str):
    try:
        cleaned = clean_text(transcript)
        formatted = format_text(cleaned)
        return formatted

    except Exception as e:
        return f"Agent Error: {str(e)}"