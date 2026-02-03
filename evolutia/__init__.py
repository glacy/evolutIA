from .evolutia_engine import EvolutiaEngine
from .variation_generator import VariationGenerator
from .llm_providers import LLMProvider, get_provider
from .exceptions import (
    EvolutiaError,
    ConfigurationError,
    ProviderError,
    ValidationError,
    MaterialExtractionError,
    ExamGenerationError,
    RAGError
)

__version__ = "0.1.1"
