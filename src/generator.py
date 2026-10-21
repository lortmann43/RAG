import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    """Generate answers using the local Qwen language model."""

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    MAX_INPUT_TOKENS = 4096

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float32,
            local_files_only=True
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
                    "You are an assistant answering questions "
                    "about the vLLM codebase. "
                    "Use only the provided context. "
                    "Answer the specific question asked. "
                    "Prefer exact identifiers, values, and API paths "
                    "when they appear in the context. "
                    "Do not confuse CLI commands with HTTP endpoints. "
                    "If the answer is not in the context, "
                    "say you do not have enough information. "
                    "Keep your answer short and precise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
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

        input_tokens = inputs["input_ids"].shape[-1]

        model_limit = getattr(
            self.model.config,
            "max_position_embeddings",
            self.MAX_INPUT_TOKENS + max_new_tokens
        )

        input_limit = min(
            self.MAX_INPUT_TOKENS,
            model_limit - max_new_tokens
        )

        if input_limit <= 0 or input_tokens > input_limit:
            raise ValueError(
                f"Prompt too long: {input_tokens} tokens. "
                f"Maximum allowed: {input_limit}. "
                "Reduce the number of retrieved sources."
            )

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
