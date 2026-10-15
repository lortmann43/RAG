from pydantic import BaseModel
from .minimal_search_answer_models import MinimalSearchResults, MinimalAnswer


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int
