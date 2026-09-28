import json
import re
from mlx_lm import load, generate

def evaluate_model(model_path, test_file="./data/valid.jsonl", num_samples=20):
    print(f"Loading {model_path}...")
    model, tokenizer = load(model_path)
    
    with open(test_file, "r") as f:
        lines = [json.loads(line) for line in f][:num_samples]
        
    valid_json_count = 0
    
    for idx, item in enumerate(lines):
        messages = item["messages"]
        system_msg = messages[0]["content"]
        user_note = messages[1]["content"]
        
        # Build prompt using Llama's official chat template
        chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_note}
        ]
        
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        
        output = generate(model, tokenizer, prompt=prompt, max_tokens=250, verbose=False)
        
        # Strip potential markdown backticks or commentary
        clean_output = re.sub(r"^```json\s*|\s*```$", "", output.strip())
        
        # Extract the JSON block if extra text surrounds it
        json_match = re.search(r"\{.*\}", clean_output, re.DOTALL)
        if json_match:
            clean_output = json_match.group(0)
            
        try:
            parsed = json.loads(clean_output)
            valid_json_count += 1
            print(f"[{idx+1}/{num_samples}] PASS: Extracted valid JSON")
        except Exception:
            print(f"[{idx+1}/{num_samples}] FAIL: Non-JSON Output\n--- Raw Output ---\n{output}\n------------------")
            
    acc = (valid_json_count / num_samples) * 100
    print(f"\n==========================================")
    print(f"Model: {model_path}")
    print(f"Valid JSON Accuracy: {acc:.1f}% ({valid_json_count}/{num_samples})")
    print(f"==========================================\n")

if __name__ == "__main__":
    evaluate_model("./fused_model")