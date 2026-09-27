from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


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
