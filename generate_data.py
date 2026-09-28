import json
import os
import random
import time
from google import genai
from google.genai import types

client = genai.Client()

MODEL_NAME = "gemini-3.5-flash-lite"
SYSTEM_PROMPT = "You are a banking compliance parsing engine. Extract structured transaction and risk metadata from compliance audit notes into strict JSON."

ALERT_TYPES = [
    "AML / BSA Velocity Flag",
    "Commercial Loan Limit Override",
    "Sanctions Screening Hold",
    "Chargeback Escalation",
    "High-Value International Wire",
    "Structuring / Smurfing Pattern",
    "Dormant Account Activation"
]

RESOLUTIONS = ["Cleared", "Escalated to Compliance", "Blocked", "Pending Documentation"]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "audit_note": {
            "type": "STRING",
            "description": "A realistic, unstructured banking compliance note with typos, codes, or narrative details."
        },
        "extracted_data": {
            "type": "OBJECT",
            "properties": {
                "transaction_type": {"type": "STRING"},
                "amount_usd": {"type": "NUMBER"},
                "flag_type": {"type": "STRING"},
                "aml_flagged": {"type": "BOOLEAN"},
                "resolution_status": {"type": "STRING"},
                "risk_severity": {"type": "STRING", "enum": ["Low", "Medium", "High"]}
            },
            "required": [
                "transaction_type",
                "amount_usd",
                "flag_type",
                "aml_flagged",
                "resolution_status",
                "risk_severity"
            ]
        }
    },
    "required": ["audit_note", "extracted_data"]
}

def generate_sample(alert_type: str, resolution: str) -> dict:
    prompt = f"Generate 1 realistic banking compliance log note and its parsed extraction. Scenario: '{alert_type}' with resolution '{resolution}'."

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.85,
        ),
    )

    data = json.loads(response.text)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data["audit_note"]},
            {"role": "assistant", "content": json.dumps(data["extracted_data"])}
        ]
    }

def main():
    target_samples = 300
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "train.jsonl")
    valid_file = os.path.join(output_dir, "valid.jsonl")

    # Reset files
    open(train_file, "w").close()
    open(valid_file, "w").close()

    print(f"Generating {target_samples} audit records using {MODEL_NAME}...\n")

    count = 0
    for i in range(target_samples):
        alert = random.choice(ALERT_TYPES)
        res = random.choice(RESOLUTIONS)
        
        try:
            sample = generate_sample(alert, res)
            
            # 80% train / 20% validation split written immediately
            target_path = train_file if random.random() < 0.8 else valid_file
            
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
                
            count += 1
            print(f"[{count}/{target_samples}] Saved to {os.path.basename(target_path)}")

            # Pause to maintain clean free-tier RPM limits
            time.sleep(1.2)

        except Exception as e:
            print(f"[{i+1}/{target_samples}] Error: {e}")
            print("Waiting 10 seconds before retrying...")
            time.sleep(10)

    print(f"\nCompleted! Saved {count} records across train.jsonl and valid.jsonl")

if __name__ == "__main__":
    main()