
from pathlib import Path

from src.models.question_models import UnansweredQuestion
from src.models.rag_dataset_model import RagDataset

dataset_dir = Path("data/datasets/UnansweredQuestions")

docs = RagDataset.model_validate_json(
    (dataset_dir / "dataset_docs_public.json").read_text()
)

code = RagDataset.model_validate_json(
    (dataset_dir / "dataset_code_public.json").read_text()
)

questions = docs.rag_questions + code.rag_questions

# Add one extra query to reach 200 retrieval requests
questions.append(
    UnansweredQuestion(question=docs.rag_questions[0].question)
)

benchmark = RagDataset(rag_questions=questions)

output_path = Path("data/output/benchmark_questions.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    benchmark.model_dump_json(indent=2),
    encoding="utf-8",
)

print(f"Created benchmark with {len(questions)} questions")
