from typing import Optional

from pydantic import BaseModel

from recipes.interfaces.config import BaseCard


class IngestCard(BaseCard):
    dataset_location: Optional[str] = None

class StepMessage(BaseModel):
    ingest: Optional[IngestCard] = None
