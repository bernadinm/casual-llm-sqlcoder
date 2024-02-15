import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

# Specify the paths to your default prompt and metadata files
DEFAULT_PROMPT_PATH = "prompt.md"
DEFAULT_METADATA_PATH = "metadata.sql"

def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def generate_prompt(question, prompt, metadata):
    formatted_prompt = prompt.format(user_question=question, table_metadata_string=metadata)
    return formatted_prompt

def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        use_cache=True,
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    return tokenizer, model

def run_inference(question, prompt=None, metadata=None):
    # Use default files if prompt or metadata are not provided
    if prompt is None:
        prompt = read_file(DEFAULT_PROMPT_PATH)
    if metadata is None:
        metadata = read_file(DEFAULT_METADATA_PATH)

    tokenizer, model = get_tokenizer_model("defog/sqlcoder-7b-2")
    formatted_prompt = generate_prompt(question, prompt, metadata)

    eos_token_id = tokenizer.eos_token_id
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False,
        num_beams=5,
    )
    generated_query = (
        pipe(
            formatted_prompt,
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )[0]["generated_text"]
        .split(";")[0]
        .split("```")[0]
        .strip()
        + ";"
    )
    return generated_query

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    question = data.get('question')
    metadata = data.get('metadata', None)  # Default to None if not provided
    prompt = data.get('prompt', None)  # Default to None if not provided

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        response = run_inference(question, prompt, metadata)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
