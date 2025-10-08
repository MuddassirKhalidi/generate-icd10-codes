import os
from openai import OpenAI
import json
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_consultation(transcript: str) -> Dict[str, Any]:
    """
    Process a medical consultation transcript and extract structured information.
    
    Args:
        transcript: The transcribed text from the audio consultation
        
    Returns:
        Dictionary containing structured medical information
    """
    
    system_prompt = """You are a medical documentation assistant. Your task is to analyze medical consultation transcripts and extract structured information.

Extract the following categories from the consultation transcript. For each category, provide a list of relevant points. If information is not mentioned, state "Not mentioned".

Categories to extract:
1. VITAL SIGNS (Temperature, Blood Pressure, Pulse Rate, Respiratory Rate, Glucose Levels)
2. CHIEF COMPLAINT
3. HISTORY OF PRESENT ILLNESS
4. PAST MEDICAL/SURGICAL HISTORY (Medical Conditions, Surgeries)
5. DRUG HISTORY AND ALLERGIES (Current Medications, Allergies)
6. FAMILY HISTORY
7. SOCIAL HISTORY
8. REVIEW OF SYSTEMS
9. PHYSICAL EXAMINATION
10. INVESTIGATIONS (including Impression/Diagnosis and Plan)
11. TREATMENT
12. PATIENT EDUCATION
13. FOLLOW-UP
14. MEDICATIONS

Return the response as a JSON object with these exact keys (in uppercase with underscores). Each key should have a list of strings as its value.

Example format:
{
    "VITAL SIGNS": ["Temperature: 98.6°F", "Blood Pressure: 120/80 mmHg"],
    "CHIEF COMPLAINT": ["Chest pain"],
    ...
}

Be concise but accurate. Only include information explicitly mentioned in the transcript."""

    user_prompt = f"""Please analyze the following medical consultation transcript and extract structured information:

Transcript:
{transcript}

Provide the extracted information in the JSON format specified."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        result["transcript"] = transcript
        
        required_keys = [
            "VITAL SIGNS",
            "CHIEF COMPLAINT",
            "HISTORY OF PRESENT ILLNESS",
            "PAST MEDICAL/SURGICAL HISTORY",
            "DRUG HISTORY AND ALLERGIES",
            "FAMILY HISTORY",
            "SOCIAL HISTORY",
            "REVIEW OF SYSTEMS",
            "PHYSICAL EXAMINATION",
            "INVESTIGATIONS",
            "TREATMENT",
            "PATIENT EDUCATION",
            "FOLLOW-UP",
            "MEDICATIONS"
        ]
        
        for key in required_keys:
            if key not in result:
                result[key] = ["Not mentioned"]
        return result
        
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing consultation: {str(e)}")


def validate_consultation_result(result: Dict[str, Any]) -> bool:
    """
    Validate that the consultation result has the expected structure.
    
    Args:
        result: The processed consultation result
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_keys = [
        "transcript",
        "VITAL SIGNS",
        "CHIEF COMPLAINT",
        "HISTORY OF PRESENT ILLNESS",
        "PAST MEDICAL/SURGICAL HISTORY",
        "DRUG HISTORY AND ALLERGIES",
        "FAMILY HISTORY",
        "SOCIAL HISTORY",
        "REVIEW OF SYSTEMS",
        "PHYSICAL EXAMINATION",
        "INVESTIGATIONS",
        "TREATMENT",
        "PATIENT EDUCATION",
        "FOLLOW-UP",
        "MEDICATIONS"
    ]
    
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing required key: {key}")
        
        if key != "transcript" and not isinstance(result[key], list):
            raise ValueError(f"Key '{key}' must be a list")
    
    return True