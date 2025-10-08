import os
from openai import OpenAI
from dotenv import load_dotenv
import io


def transcribe_audio(audio_content: bytes, filename: str) -> str:
    """
    Transcribe Arabic-English mixed audio without translation.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    audio_file = io.BytesIO(audio_content)
    audio_file.name = filename  
    
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    
    return transcript.text
