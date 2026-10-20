import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    """Generate answers using the local Qwen language model."""

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float32
        )
        self.model.eval()

    def generate(
            self,
            question: str,
            context: str,
            max_new_tokens: int = 150
    ) -> str:
        """Generate an answer grounded in the provided context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer questions using only the provided context. "
                    "If the context does not contain the answer, "
                    "say that you do not have enough information. "
                    "Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )
