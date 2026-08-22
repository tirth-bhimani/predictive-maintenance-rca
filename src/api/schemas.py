from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):

    features: List[float]


class PredictionResponse(BaseModel):

    RUL: float

    failing_soon: bool

    top_factors: List[dict]

    root_cause: dict

    recommendation: dict