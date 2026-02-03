"""
Excepciones personalizadas para EvolutIA.
Define una jerarquía de excepciones para manejo consistente de errores.
"""

class EvolutiaError(Exception):
    """Excepción base para todos los errores de EvolutIA."""
    pass


class ConfigurationError(EvolutiaError):
    """Error en la configuración del sistema."""
    pass


class ProviderError(EvolutiaError):
    """Error en el proveedor de LLM."""
    pass


class ValidationError(EvolutiaError):
    """Error de validación de datos."""
    pass


class MaterialExtractionError(EvolutiaError):
    """Error al extraer materiales didácticos."""
    pass


class ExamGenerationError(EvolutiaError):
    """Error al generar examen."""
    pass


class RAGError(EvolutiaError):
    """Error en el sistema RAG."""
    pass
