from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superflous:str = Field(description="Critque of what is superflous.")

class AnswerQuestion(BaseModel):
    """Answer the  questions"""

    answer: str = Field(description="~250 word detailed answer to the question.")
    refection : Reflection = Field(description="Your Reflection on the initial answer")
    search_queries:List[str] = Field(
        description="1-3 search Queries for researching improvements to address the critique of your current answer."
    ) 

class ReviseAnswer(BaseModel):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citation motivating your updated answer."
        
    )