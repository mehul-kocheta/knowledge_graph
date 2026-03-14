from ollama import client as ollama_client


class OllamaClient:
    def generate(self, model_name: str, prompt: str, system: str | None = None) -> str | None:
        response, _ = ollama_client.generate(model_name=model_name, prompt=prompt, system=system)
        return response
