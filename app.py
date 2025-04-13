from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = Flask(__name__)

# Load model and tokenizer
cache_dir = "./model_cache"
model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

@app.route('/extract-fbs', methods=['POST'])
def extract_fbs():
    try:
        data = request.json
        text = data.get("text", "")

        prompt = f"Extract the fasting blood sugar level (numeric value and unit, e.g., 124 mg/dl) from the following medical report:\n{text}\nAnswer:"
        response = pipe(prompt, max_new_tokens=200)
        result = response[0]["generated_text"].split("Answer:")[-1].strip()

        return jsonify({
            "fasting_blood_sugar": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port, debug=True)
