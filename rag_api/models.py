from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, List


status_type = Literal['online', 'offline']

class StateResponse(BaseModel):
    api_status: status_type = Field(description='State of API')
    startup_time: datetime = Field(description='API startup datetime')
    open_ai_status: status_type = Field(description='State of Open AI API')
    content_files: List[str] = Field(description='List of loaded data files (knowledge base)')
    model_name: str = Field(description='Type and name of chat model')

class LLMResponse(BaseModel):
    answer: str = Field(description='Response of LLM model')

class LLMRequest(BaseModel):
    query: str = Field(description='Query to LLM model (requested question)')
    temperature: float = Field(default=0.7, description='Optional generative parameter - temperature (Sampling temperature (creativity))')
    top_p: float = Field(default=0.9, description='Optional generative parameter - Top-p (Nucleus sampling probability)')
    top_k: int = Field(default=3, description='Optional generative parameter - Top-k (Number of knowledge snippets to include)')