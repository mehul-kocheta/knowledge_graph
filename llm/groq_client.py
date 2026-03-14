import os
from groq import Groq


class GroqClient:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self._client = Groq(api_key=api_key)

    def generate(self, model_name: str, prompt: str, system: str | None = None) -> str | None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content
