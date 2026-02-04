from pydantic import BaseModel

class StaticResponseNodeConfig(BaseModel):
    """Configuration for the Static Response Node."""
    enabled: bool = True