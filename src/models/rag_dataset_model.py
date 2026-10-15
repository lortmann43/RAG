from pydantic import BaseModel
from .question_models import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
