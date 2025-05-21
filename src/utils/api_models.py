import os
from dotenv import load_dotenv

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

load_dotenv()


def get_model(branch: str = "openai"):
    if branch == "openai":
        model_name = os.getenv('OPENAI_MODEL_NAME', 'gpt-4o')
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('OPENAI_API_KEY', 'no-api-key-provided')

        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url=base_url,
                api_key=api_key
            )
        )
    if branch == "google":
        model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
        api_key = os.getenv('GEMINI_API_KEY', 'no-api-key-provided')
        return GeminiModel(
            model_name,
            provider=GoogleGLAProvider(api_key=api_key)
        )
    return None
