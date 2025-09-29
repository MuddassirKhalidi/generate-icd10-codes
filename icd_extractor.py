# id_extractor.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from embedding_en import search_icd10
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_icd10(consultation: str):
    """
    Extract ICD-10 diagnoses using GPT with reasoning and explanation.
    The model explains its choices before providing structured output.
    """
    prompt = f"""
You are an expert medical coding assistant specializing in ICD-10 diagnosis extraction.

Guidelines:
- Always follow the diagnosis described in the consultation.
- Do NOT assume a more severe or acute variant based on risk factors or symptoms.
- Use COMPLETE official ICD-10 diagnosis descriptions, including ALL specificity qualifiers.
- For example: Use "Type 2 diabetes mellitus without complications" NOT just "Diabetes"
- For example: Use "Acute bronchitis, unspecified" NOT just "Bronchitis"
- For example: Use "Gastroesophageal reflux disease without esophagitis" NOT just "GERD"
- If the consultation provides specific details (laterality, type, severity), include them in the description.
- If details are insufficient, ALWAYS use "unspecified" or equivalent variants.
- Include confirmed diagnoses, comorbidities, past history, family history (Z-codes), and explicit risk factors.

Instructions:
1. First, analyze the consultation and explain your reasoning for each diagnosis you identify.
2. Discuss why you chose specific descriptions and how they align with the clinical information.
3. Explain your specificity choices (why "unspecified" vs. more specific variants).
4. Defend your choices by referencing relevant details from the consultation.
5. After your explanation, provide the final list of COMPLETE ICD-10 diagnosis descriptions.

Output Format:
First, write your analysis and reasoning in natural language.

Then, end your response with:

<diagnoses>
Complete ICD-10 diagnosis description (with specificity)
Complete ICD-10 diagnosis description (with specificity)
</diagnoses>

Consultation to analyze:
{consultation}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.choices[0].message.content

    # Extract <diagnoses> section
    try:
        if "<diagnoses>" in response_text and "</diagnoses>" in response_text:
            start_idx = response_text.find("<diagnoses>") + len("<diagnoses>")
            end_idx = response_text.find("</diagnoses>")
            diagnoses_text = response_text[start_idx:end_idx].strip()
        else:
            # Fallback: treat entire response as diagnoses
            diagnoses_text = response_text.strip()

        diagnoses_list = []
        for line in diagnoses_text.splitlines():
            line = line.strip()
            # Remove any leading bullets or numbers
            if line.startswith(("-", "*", "•")):
                line = line[1:].strip()
            elif line and line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[1].strip()
            
            if line:
                diagnoses_list.append({"description": line, "code": None})

        return diagnoses_list

    except Exception:
        lines = response_text.strip().splitlines()
        return [{"description": line.strip(), "code": None} for line in lines if line.strip()]


def extract_icd10_with_validation(consultation: str, top_k: int = 1, threshold: float = 0.6):
    """
    Validate GPT outputs using search_icd10 to ensure ICD-10 descriptions exist.
    For each extracted description, search for it and return the highest match.
    
    API remains the same for compatibility with existing components.
    """
    # Step 1: Extract ICD descriptions using LLM (with reasoning)
    diagnoses = extract_icd10(consultation)
    results = []

    # Step 2: For each extracted description, search and return highest pick
    for diag in diagnoses:
        # Search for the diagnosis description
        codes, scores, descriptions = search_icd10(diag["description"], top_k=top_k, verbose=True)
        
        # Find matches above threshold
        valid_matches = []
        for code, score, desc in zip(codes, scores, descriptions):
            if score >= threshold:
                valid_matches.append({
                    "code": code,
                    "description": desc,
                    "score": float(score)
                })

        # Use the highest match if available
        if valid_matches:
            results.append({
                "diagnosis": valid_matches[0]["description"],
                "matches": valid_matches
            })
        else:
            # No valid match found - include original description with empty matches
            results.append({
                "diagnosis": diag["description"],
                "matches": []
            })

    return results


def extract_icd10_advanced(consultation: str):
    """
    Return a simple list of diagnoses with codes inline.
    """
    validated = extract_icd10_with_validation(consultation)
    final_list = []
    for item in validated:
        if item["matches"]:
            match = item["matches"][0]
            final_list.append(f"{match['description']} ({match['code']})")
        else:
            final_list.append(item["diagnosis"])
    return final_list