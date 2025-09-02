import json
import csv
import os

def read_icd10_from_file(filename):
    codes = []
    descriptions = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            
            # Split by whitespace; first token is code, rest is description
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                # skip malformed lines without description
                continue
            
            code, description = parts[0], parts[1]
            codes.append(code)
            descriptions[code] = description

    return codes, descriptions

def save_icd10_from_file(filename):
    os.makedirs('archive/icd10data', exist_ok=True)

    codes, descriptions = read_icd10_from_file(filename)

    if not codes:
        print("\033[91mERROR: No codes found in file.\033[0m")
        return

    with open('archive/icd10data/icd10_codes.json', 'w', encoding='utf-8') as json_file:
        json.dump(codes, json_file, indent=4)

    with open('archive/icd10data/icd10_descriptions.json', 'w', encoding='utf-8') as json_file:
        json.dump(descriptions, json_file, indent=4)

    with open('archive/icd10data/icd10_descriptions.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Code', 'Description'])
        for code, description in descriptions.items():
            writer.writerow([code, description])

    print(f"\033[92mSUCCESS: {len(codes)} ICD-10 codes saved from file.\033[0m")


import json

# Load the dict from the JSON file
with open('archive/icd10data/icd10_descriptions.json', 'r') as f:
    data = json.load(f)

# Reverse keys and values
reversed_data = {v: k for k, v in data.items()}

# Dump the reversed dict back to the same JSON file
with open('archive/icd10data/icd10_descriptions.json', 'w') as f:
    json.dump(reversed_data, f, indent=4)

print(f"\033[92mSUCCESS\033[0m")
