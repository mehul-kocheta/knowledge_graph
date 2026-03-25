from typing import Protocol


class LLMClient(Protocol):
    def generate(self, model_name: str, prompt: str, system: str   = None) -> str  :
        ...
