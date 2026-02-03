# Changelog

Todas las variaciones notables de este proyecto serán documentadas en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-02

### Optimizado
- **Caché LLM**: Implementado write-behind con debounce (5 segundos) para persistencia a disco
  - Reduce dramáticamente el I/O de disco (antes: cada put(), ahora: solo después de inactividad)
  - Agregado límite de memoria configurable (default: 500MB) con tracking de uso por entrada
  - Implementado eviction automático basado en RAM cuando se excede el límite
  - Estadísticas expandidas: `memory_mb`, `memory_limit_mb`, `memory_usage_percent`
- **MaterialExtractor**: Implementado caché de rutas de archivos válidos
  - Almacena metadatos de archivos procesados para evitar escaneos repetidos del filesystem
  - TTL de 5 minutos para invalidar caché automáticamente
  - Verificación de modificación de archivos usando timestamps
  - Método `clear_cache()` para limpiar manualmente el caché
  - Método `get_cache_stats()` para obtener estadísticas del caché
- **RAGIndexer/RAGRetriever**: Implementado lazy loading de modelos de embeddings
  - Modelos de embeddings solo se cargan cuando se necesitan (no en __init__)
  - Reduce el tiempo de inicialización y el uso de memoria cuando RAG no se usa
  - Método `_ensure_embeddings_initialized()` para inicialización bajo demanda
- **ChromaDB queries**: Agregado límite de paginación para evitar cargar colecciones completas
  - Parámetro `max_results_limit` en configuración de retrieval (default: 100)
  - Limita el número máximo de resultados en queries de ChromaDB
  - Aplicado a `retrieve_similar_exercises()` y `hybrid_search()`
- **Proveedores asíncronos**: Agregado manejo de errores con retry automático
  - Decorador `@retry_async` con backoff exponencial (1s → 10s)
  - Máximo 3 reintentos por defecto, configurable
  - Logging de cada intento fallido con información del error
  - Clase `CircuitBreaker` para evitar llamadas a servicios fallidos
  - Decorador `@with_circuit_breaker` para proteger funciones
- **Eliminado directorio build/ del control de versiones**:
  - Agregado `build/` a `.gitignore`
  - Eliminadas ~12k líneas duplicadas del repositorio
  - Reducción del tamaño del repositorio
- **Imports centralizados**: Nuevo módulo `evolutia/imports.py`
  - Clase `OptionalImports` para gestionar imports de dependencias opcionales
  - Centralizados imports de ChromaDB, sentence-transformers, OpenAI, Anthropic, Gemini
  - Reducción de duplicación de código try/except por todo el proyecto
- **Soporte asíncrono para llamadas LLM**: Nuevo módulo `evolutia/async_llm_providers.py`
  - Wrappers asíncronos para proveedores LLM (`AsyncOpenAIProvider`, `AsyncAnthropicProvider`, `AsyncGeminiProvider`)
  - Nuevo método `generate_variations_async()` en `EvolutiaEngine` usando `asyncio.gather()`
  - Uso de `asyncio.Semaphore` para limitar concurrencia
  - Más eficiente que `ThreadPoolExecutor` para operaciones I/O-bound
  - Mantenida compatibilidad con versión síncrona existente

### Añadido
- **Nuevo módulo**: `evolutia/retry_utils.py`
  - Decoradores `@retry_async` y `@retry_sync` para reintentos automáticos
  - Clase `CircuitBreaker` para implementar patrón Circuit Breaker
  - Backoff exponencial configurable
  - Callbacks opcionales antes de cada reintento

### Documentación
- **README.md**: Reescrito completamente con enfoque de storytelling para usuarios finales
  - Narrativa enganchante desde el problema del profesor hasta la solución
  - Tabla comparativa "Sin EvolutIA vs Con EvolutIA"
  - Historias de uso reales con ejemplos concretos
  - Instalación paso a paso con 4 etapas claras
  - Sección de optimizaciones simplificada y orientada a beneficios
  - Sección "Herramienta de IA Asistente: opencode" documentando el modelo GLM-4.7
  - Total: 500 líneas (era 297) con tono más accesible
- **CHANGELOG.md**: Documentación completa de todas las 9 optimizaciones implementadas
- **AGENTS.md**: Actualizado con nuevos módulos (cache mejorado, retry_utils)
- **docs/ARCHITECTURE.md**: Secciones nuevas para caché LLM, proveedores async, imports centralizados, optimizaciones adicionales

### Mantenimiento
- Migración del paquete `google.generativeai` a `google.genai` para resolver advertencias de deprecación (FutureWarning) y asegurar compatibilidad futura.

## [Unreleased]

### Optimizado
- **Caché LLM**: Implementado write-behind con debounce (5 segundos) para persistencia a disco
  - Reduce dramáticamente el I/O de disco (antes: cada put(), ahora: solo después de inactividad)
  - Agregado límite de memoria configurable (default: 500MB) con tracking de uso por entrada
  - Implementado eviction automático basado en RAM cuando se excede el límite
  - Estadísticas expandidas: `memory_mb`, `memory_limit_mb`, `memory_usage_percent`
- **MaterialExtractor**: Implementado caché de rutas de archivos válidos
  - Almacena metadatos de archivos procesados para evitar escaneos repetidos del filesystem
  - TTL de 5 minutos para invalidar caché automáticamente
  - Verificación de modificación de archivos usando timestamps
  - Método `clear_cache()` para limpiar manualmente el caché
  - Método `get_cache_stats()` para obtener estadísticas del caché
- **RAGIndexer/RAGRetriever**: Implementado lazy loading de modelos de embeddings
  - Modelos de embeddings solo se cargan cuando se necesitan (no en __init__)
  - Reduce el tiempo de inicialización y el uso de memoria cuando RAG no se usa
  - Método `_ensure_embeddings_initialized()` para inicialización bajo demanda
- **ChromaDB queries**: Agregado límite de paginación para evitar cargar colecciones completas
  - Parámetro `max_results_limit` en configuración de retrieval (default: 100)
  - Limita el número máximo de resultados en queries de ChromaDB
  - Aplicado a `retrieve_similar_exercises()` y `hybrid_search()`
- **Proveedores asíncronos**: Agregado manejo de errores con retry automático
  - Decorador `@retry_async` con backoff exponencial (1s → 10s)
  - Máximo 3 reintentos por defecto, configurable
  - Logging de cada intento fallido con información del error
  - Clase `CircuitBreaker` para evitar llamadas a servicios fallidos
  - Decorador `@with_circuit_breaker` para proteger funciones
- **Eliminado directorio build/ del control de versiones**:
  - Agregado `build/` a `.gitignore`
  - Eliminadas ~12k líneas duplicadas del repositorio
  - Reducción del tamaño del repositorio
- **Imports centralizados**: Nuevo módulo `evolutia/imports.py`
  - Clase `OptionalImports` para gestionar imports de dependencias opcionales
  - Centralizados imports de ChromaDB, sentence-transformers, OpenAI, Anthropic, Gemini
  - Reducción de duplicación de código try/except por todo el proyecto
- **Soporte asíncrono para llamadas LLM**: Nuevo módulo `evolutia/async_llm_providers.py`
  - Wrappers asíncronos para proveedores LLM (`AsyncOpenAIProvider`, `AsyncAnthropicProvider`, `AsyncGeminiProvider`)
  - Nuevo método `generate_variations_async()` en `EvolutiaEngine` usando `asyncio.gather()`
  - Uso de `asyncio.Semaphore` para limitar concurrencia
  - Más eficiente que `ThreadPoolExecutor` para operaciones I/O-bound
  - Mantenida compatibilidad con versión síncrona existente

### Añadido
- **Nuevo módulo**: `evolutia/retry_utils.py`
  - Decoradores `@retry_async` y `@retry_sync` para reintentos automáticos
  - Clase `CircuitBreaker` para implementar patrón Circuit Breaker
  - Backoff exponencial configurable
  - Callbacks opcionales antes de cada reintento

### Mantenimiento
- Migración del paquete `google.generativeai` a `google.genai` para resolver advertencias de deprecación (FutureWarning) y asegurar compatibilidad futura.
- Refactorización de `evolutia/llm_providers.py` para eliminar duplicación de código:
  - Nueva clase base `OpenAICompatibleProvider` con lógica compartida para proveedores compatibles con OpenAI API
  - Reducción de 54 líneas de código (16%) en el módulo de proveedores
  - Mejora de mantenibilidad al centralizar la lógica de configuración de cliente y generación de contenido
  - Proveedores refactorizados: `OpenAIProvider`, `LocalProvider`, `DeepSeekProvider`, `GenericProvider`
- Estandarización de type hints en módulos principales:
  - `evolutia_engine.py`: Mejora de tipos en métodos como `_generate_single_variation()`, `_generate_creation_mode()`, `generate_variations_parallel()`, `generate_exam_files()`
  - `complexity_validator.py`: Mejora de tipos en `validate_batch()` (antes `list` genérico, ahora `List[Tuple[Dict, Dict, Dict]]`)
  - `config_manager.py`: Mejora de tipos en `__init__()` (acepta ahora `Union[Path, str]`)
  - `variation_generator.py`: Mejora de tipos en `generate_variation()`, `generate_new_exercise_from_topic()`, `_create_quiz_prompt()`
  - `material_extractor.py`: Mejora de tipos en `__init__()` (acepta ahora `Union[Path, str]`)
  - `exam_generator.py`: Mejora de tipos en `__init__()`, `generate_exam_frontmatter()` (usa `Optional[List[str]]`)
  - `exercise_analyzer.py`: Mejora de tipos en `analyze()` (ahora `Dict[str, Optional[str | int | float | List[str]]`)
  - `llm_providers.py`: Añadido `Union` a imports para mejor flexibilidad de tipos
- Unificación de política de manejo de errores:
  - Nueva jerarquía de excepciones personalizadas (`EvolutiaError`, `ConfigurationError`, `ProviderError`, `ValidationError`, `MaterialExtractionError`, `ExamGenerationError`, `RAGError`)
  - Mejora de mensajes de logging con contexto consistente (nombre del componente, información relevante)
  - Aplicación de políticas de fail-fast vs graceful degradation de manera consistente
  - Mejor documentación de retorno de métodos (docstrings actualizados)
  - Logging mejorado con prefijos de componente para facilitar diagnóstico
  - Archivos mejorados: `llm_providers.py`, `variation_generator.py`, `evolutia_engine.py`, `material_extractor.py`, `complexity_validator.py`, `exam_generator.py`
   - Nueva documentación: `docs/ERROR_HANDLING.md` con política completa y ejemplos
- Implementación de validación exhaustiva de inputs:
  - Nuevo módulo `evolutia/validation/args_validator.py` con 19 métodos de validación para argumentos CLI
  - Nuevo módulo `evolutia/validation/config_validator.py` con 25 métodos de validación para configuración
  - Validación automática de argumentos CLI antes de ejecutar el engine
  - Validación automática de configuración al cargar `evolutia_config.yaml`
  - Validaciones de rutas, tipos numéricos, valores de enum, rangos válidos, combinaciones de argumentos
  - Sistema de errores (bloqueantes) y warnings (no bloqueantes pero informativos)
   - Nueva documentación: `docs/VALIDATION.md` con guía completa de validación
   - 40 tests nuevos para los validadores (19 para ArgsValidator, 21 para ConfigValidator)
- Implementación de caché de LLM para ahorro de costos:
  - Integrado `LLMCache` en todos los proveedores LLM
  - Añadido parámetro `cache` opcional a `LLMProvider`, `OpenAICompatibleProvider`, `AnthropicProvider`, `GeminiProvider`, `LocalProvider`, `DeepSeekProvider`, `GenericProvider`
  - Implementado lógica de get/put caché en `_openai_generate_content()`, `AnthropicProvider.generate_content()`, `GeminiProvider.generate_content()`
  - Filtrado de respuestas vacías, de error y muy cortas
  - TTL configurable y LRU eviction para expiración automática
  - Metadata de caché (provider, model, temperature, max_tokens)
  - 13 tests nuevos para `LLMCache` y `ExerciseAnalysisCache`

### Añadido
- Nuevo argumento CLI `--analyze` para auto-descubrimiento de configuración y estructura del proyecto.

### Cambiado
- Reescritura integral del `README.md`: unificación de instrucciones de instalación, simplificación de guías de uso y eliminación de redundancias.
- La guía de submódulos `GUIDE_SUBMODULES.md` ha sido archivada en `docs/legacy/` ya que el método recomendado de instalación es vía pip.

## [0.1.1]2026-01-31

### Corregido
- Solucionado error `AttributeError: 'VariationGenerator' object has no attribute 'generate_variation'`. Este método faltaba en la distribución 0.1.0 y causaba fallos en el modo de variación.
- Añadido argumento CLI `--analyze` faltante en la distribución anterior.

## [0.1.0] - 2026-01-30

### Añadido
- Primera versión pública en PyPI (`pip install evolutia`).
- Sistema completo de generación de exámenes basado en IA (OpenAI, Anthropic, Gemini).
- Soporte para RAG (Retrieval-Augmented Generation) para enriquecer el contexto.
- Modos de variación ("variation") y creación ("creation") de ejercicios.
- Integración con materiales didácticos en formato Markdown/MyST.
- Soporte para ejecución local con Ollama/LM Studio.
- Documentación completa y guías de uso.
