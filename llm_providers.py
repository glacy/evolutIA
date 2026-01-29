"""
Módulo que define los proveedores de LLM abstractos y concretos.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Clase base abstracta para proveedores de LLM."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.api_key = self._get_api_key()
        if self.api_key:
            self._setup_client()
        else:
             # Some providers like local might not strictly need an API key from env
             pass

    @abstractmethod
    def _get_api_key(self) -> Optional[str]:
        """Obtiene la API key de las variables de entorno."""
        pass

    @abstractmethod
    def _setup_client(self):
        """Configura el cliente de la API."""
        pass

    @abstractmethod
    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        """Genera contenido a partir de un prompt."""
        pass


class OpenAIProvider(LLMProvider):
    """Proveedor para OpenAI."""

    def _get_api_key(self) -> Optional[str]:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            logger.warning("OPENAI_API_KEY no encontrada")
        return key

    def _setup_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("Biblioteca openai no instalada. Instala con: pip install openai")
            self.client = None

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.client: return None
        
        system_content = system_prompt or "Eres un experto en métodos matemáticos para física e ingeniería."
        model = self.model_name or "gpt-4"
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error llamando a OpenAI API: {e}")
            return None


class AnthropicProvider(LLMProvider):
    """Proveedor para Anthropic (Claude)."""

    def _get_api_key(self) -> Optional[str]:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("ANTHROPIC_API_KEY no encontrada")
        return key

    def _setup_client(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.error("Biblioteca anthropic no instalada. Instala con: pip install anthropic")
            self.client = None

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.client: return None

        system_content = system_prompt or "Eres un experto en métodos matemáticos para física e ingeniería."
        model = self.model_name or "claude-3-opus-20240229"

        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.7),
                system=system_content,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error llamando a Anthropic API: {e}")
            return None


class GeminiProvider(LLMProvider):
    """Proveedor para Google Gemini."""

    def _get_api_key(self) -> Optional[str]:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            logger.warning("GOOGLE_API_KEY no encontrada")
        return key

    def _setup_client(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            logger.error("Biblioteca google-generativeai no instalada")
            self.genai = None

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.genai: return None

        model_name = self.model_name or "gemini-2.5-pro"
        if model_name == 'gemini': model_name = "gemini-2.5-pro"

        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": kwargs.get("max_tokens", 8192),
            "response_mime_type": "text/plain",
        }

        try:
            model_instance = self.genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                # System instructions can be passed to model if supported, 
                # or prepended to prompt. Gemini 1.5 supports system_instruction arg.
                system_instruction=system_prompt
            )
            response = model_instance.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error llamando a Gemini API: {e}")
            return None


class LocalProvider(LLMProvider):
    """Proveedor para modelos locales (Ollama/LM Studio) vía OpenAI compatible API."""

    def __init__(self, model_name: Optional[str] = None, base_url: str = "http://localhost:11434/v1"):
        self.base_url = base_url
        super().__init__(model_name)

    def _get_api_key(self) -> Optional[str]:
        return "not-needed"

    def _setup_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=300.0
            )
        except ImportError:
            logger.error("Biblioteca openai no instalada")
            self.client = None

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.client: return None

        system_content = system_prompt or "Eres un experto en métodos matemáticos para física e ingeniería."
        model = self.model_name or "llama3"

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error llamando a Local API: {e}")
            return None


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory method para obtener un proveedor."""
    if provider_name == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_name == "anthropic":
        return AnthropicProvider(**kwargs)
    elif provider_name == "gemini":
        return GeminiProvider(**kwargs)
    elif provider_name == "local":
        return LocalProvider(**kwargs)
    else:
        raise ValueError(f"Proveedor desconocido: {provider_name}")
