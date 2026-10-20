
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)
model.eval()

messages = [
    {
        "role": "user",
        "content": "Explain RAG in one sentence.",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
)

print("Generating answer...")

with torch.inference_mode():
    output = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_tokens = output[0][inputs.shape[-1]:]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\nAnswer:")
print(answer)
