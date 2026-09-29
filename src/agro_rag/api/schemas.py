from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    database: str
    cache: str
    llm: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500, examples=["como calcular a dose de calcário?"])
    k: int = Field(default=5, ge=1, le=20)
    category: str | None = Field(default=None, examples=["solo"])
    collection: str | None = None


class SearchHit(BaseModel):
    source: str
    title: str
    category: str
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    took_ms: float


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=500, examples=["quando aplicar nitrogênio em cobertura no milho?"]
    )
    k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    source: str
    title: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    took_ms: float


class AgentResponse(BaseModel):
    answer: str
    sources: list[Source]
    steps: list[str]
    took_ms: float
