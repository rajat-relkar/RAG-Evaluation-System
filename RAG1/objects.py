from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str = Field(description="the answwr that is needed for the user question")
