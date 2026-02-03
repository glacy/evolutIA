# Arquitectura Interna

Este documento describe la arquitectura interna de EvolutIA para desarrolladores y contribuidores.

## Estructura del Módulo

```
evolutia/
 ├── __init__.py
 ├── evolutia_engine.py       # Orquestador principal
 ├── variation_generator.py    # Generación de ejercicios
 ├── exercise_analyzer.py     # Análisis de complejidad
 ├── complexity_validator.py  # Validación de variaciones
 ├── material_extractor.py    # Extracción de materiales
 ├── exam_generator.py        # Generación de archivos
 ├── config_manager.py        # Gestión de configuración
 ├── llm_providers.py         # Abstracción de proveedores LLM
 ├── async_llm_providers.py   # Proveedores asíncronos de LLM
 ├── imports.py               # Imports centralizados
 └── utils/
     ├── json_parser.py       # Parseo robusto de JSON
     ├── markdown_parser.py   # Parseo de Markdown/MyST
     └── math_extractor.py    # Extracción de matemáticas
 └── rag/                    # Sistema RAG
     ├── rag_manager.py       # Gestor RAG
     ├── rag_indexer.py       # Indexación
     ├── rag_retriever.py     # Recuperación
     └── ...
 └── cache/                  # Sistema de caché
     ├── llm_cache.py        # Caché de respuestas LLM
     └── exercise_cache.py   # Caché de análisis de ejercicios
 └── validation/             # Sistema de validación
     ├── args_validator.py   # Validación de argumentos CLI
     └── config_validator.py # Validación de configuración
```

## Sistema de Proveedores LLM

El módulo `llm_providers.py` implementa una jerarquía de clases para soportar múltiples proveedores de IA.

### Jerarquía de Clases

```
LLMProvider (ABC)
├── OpenAICompatibleProvider
│   ├── OpenAIProvider
│   ├── LocalProvider (Ollama/LM Studio)
│   ├── DeepSeekProvider
│   └── GenericProvider (Groq, Mistral, etc.)
├── AnthropicProvider
└── GeminiProvider
```

### Clases Principales

#### `LLMProvider` (Clase Base Abstracta)

Clase base abstracta que define la interfaz común para todos los proveedores.

**Constantes:**
- `DEFAULT_SYSTEM_PROMPT`: Prompt del sistema por defecto
- `DEFAULT_MAX_TOKENS`: 2000
- `DEFAULT_TEMPERATURE`: 0.7

**Métodos Abstractos:**
- `_get_api_key()`: Obtiene la API key de variables de entorno
- `_setup_client()`: Configura el cliente de la API
- `generate_content()`: Genera contenido a partir de un prompt

#### `OpenAICompatibleProvider` (Clase Intermedia)

Base clase para proveedores compatibles con OpenAI API. Implementa lógica compartida para:
- Configuración de cliente OpenAI
- Generación de contenido usando el formato de chat completions
- Manejo de errores y logging

**Métodos Implementados:**
- `_setup_openai_client()`: Configura cliente OpenAI con soporte para `base_url` y `timeout`
- `_openai_generate_content()`: Implementa la lógica de generación compartida

**Ventajas:**
- Evita duplicación de código entre proveedores similares
- Facilita agregar nuevos proveedores OpenAI-compatible
- Centraliza el manejo de errores y parámetros por defecto

**Proveedores que heredan:**
- `OpenAIProvider`: OpenAI API oficial
- `LocalProvider`: Modelos locales vía Ollama/LM Studio
- `DeepSeekProvider`: DeepSeek API (compatible OpenAI)
- `GenericProvider`: Cualquier API OpenAI-compatible (Groq, Mistral, etc.)

### Ejemplo: Agregar un Nuevo Proveedor OpenAI-Compatible

Para agregar un nuevo proveedor compatible con OpenAI API:

```python
class NuevoProvider(OpenAICompatibleProvider):
    """Proveedor para Nuevo Servicio (OpenAI-compatible)."""

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name, base_url="https://api.nuevoservicio.com")

    def _get_api_key(self) -> Optional[str]:
        key = os.getenv("NUEVO_API_KEY")
        if not key:
            logger.warning("NUEVO_API_KEY no encontrada")
        return key

    def _setup_client(self):
        self._setup_openai_client(self.api_key, base_url=self.base_url)

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        return self._openai_generate_content("NuevoServicio", "modelo-por-defecto", system_prompt, prompt=prompt, **kwargs)
```

Luego actualizar el factory method `get_provider()`:

```python
def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    if provider_name == "nuevo":
        return NuevoProvider(**kwargs)
    # ... otros proveedores
```

## Flujo de Ejecución Principal

### 1. Extracción de Materiales
```
CLI → EvolutiaEngine → MaterialExtractor
   ↓
   ↓ Lee archivos .md de temas
   ↓
   ↓ Extrae frontmatter, ejercicios, soluciones
   ↓
```

### 2. Análisis de Ejercicios
```
EvolutiaEngine → ExerciseAnalyzer
   ↓
   ↓ Identifica tipo (demostración/cálculo/aplicación)
   ↓
   ↓ Cuenta pasos de solución
   ↓
   ↓ Extrae conceptos y variables
   ↓
   ↓ Calcula complejidad matemática
   ↓
```

### 3. Generación de Variaciones
```
EvolutiaEngine → VariationGenerator → LLMProvider
   ↓
   ↓ Crea prompt contextualizado
   ↓
   ↓ Llama a API del proveedor
   ↓
   ↓ Parsea respuesta (JSON o Markdown)
   ↓
```

### 4. Validación de Complejidad
```
EvolutiaEngine → ComplexityValidator
   ↓
   ↓ Compara métricas original vs variación
   ↓
   ↓ Verifica aumento de complejidad
   ↓
   ↓ Acepta o rechaza variación
   ↓
```

### 5. Generación de Archivos
```
EvolutiaEngine → ExamGenerator
   ↓
   ↓ Crea estructura del examen
   ↓
   ↓ Genera archivos individuales (ejercicios y soluciones)
   ↓
   ↓ Escribe metadatos y frontmatter
   ↓
```

## Sistema RAG (Opcional)

Cuando está habilitado (`--use_rag`), el flujo incluye:

1. **Indexación**: `RAGIndexer` procesa materiales y crea embeddings
2. **Recuperación**: `RAGRetriever` busca contexto relevante
3. **Enriquecimiento**: `EnhancedVariationGenerator` usa contexto para generar ejercicios más alineados

## Configuración y Estado

El sistema maneja configuración en múltiples niveles (precedencia de mayor a menor):

1. **Argumentos CLI**: Prioridad más alta
2. **Archivo `evolutia_config.yaml`**: Configuración del proyecto
3. **Variables de entorno (`.env`)**: API keys
4. **Defaults del código**: Valores por defecto internos

## Patrones de Diseño Utilizados

- **Factory Method**: `get_provider()` para crear instancias de proveedores
- **Template Method**: `LLMProvider` define el flujo, subclasses implementan pasos específicos
- **Strategy Pattern**: Diferentes proveedores LLM como estrategias intercambiables
- **Facade**: `EvolutiaEngine` como fachada que orquesta el flujo completo

## Optimizaciones Recientes

### Refactorización de llm_providers.py (v0.2.0)

**Objetivo**: Eliminar duplicación de código entre proveedores OpenAI-compatible

**Cambios**:
- Creación de `OpenAICompatibleProvider` como clase intermedia
- Extracción de métodos comunes: `_setup_openai_client()` y `_openai_generate_content()`
- Reducción de código: 319 → 265 líneas (-16%)
- Proveedores refactorizados: OpenAI, Local, DeepSeek, Generic

**Beneficios**:
- Mejor mantenibilidad: cambios en lógica compartida solo en un lugar
- Fácil extensión: agregar nuevos proveedores OpenAI-compatible requiere menos código
- Menor superficie de bugs: menos código duplicado que mantener sincronizado

### Estandarización de Type Hints (v0.2.0)

**Objetivo**: Mejorar la calidad del código y facilitar el mantenimiento mediante type hints completos

**Cambios**:
- `evolutia_engine.py`:
  - Mejora en `_generate_single_variation()`: tipos explícitos para `generator`, `validator`, `exercise_base`, `analysis`, `args` (ahora `argparse.Namespace`)
  - Mejora en `_generate_creation_mode()`: tipos para `generator`, `topic`, `tags` (ahora `List[str]`), `complexity`, `ex_type`
  - Mejora en `generate_variations_parallel()`: tipo para `args` (ahora `argparse.Namespace`)
  - Mejora en `generate_exam_files()`: tipo para `output_dir` (ahora `Union[Path, str]`)
- `complexity_validator.py`:
  - Mejora en `validate_batch()`: ahora usa `List[Tuple[Dict, Dict, Dict]]` en lugar de `list` genérico
- `config_manager.py`:
  - Mejora en `__init__()`: acepta `Union[Path, str]` para `base_path` y `config_path`
- `variation_generator.py`:
  - Mejora en `generate_variation()`: tipos para `exercise` (ahora `Dict[str, Any]`) y `analysis` (ahora `Dict[str, Any]`)
  - Mejora en `generate_new_exercise_from_topic()`: tipo para `tags` (ahora `Optional[List[str]]`)
  - Mejora en `_create_quiz_prompt()`: tipo para `context_info` (ahora `Dict[str, Any]`)
- `material_extractor.py`:
  - Mejora en `__init__()`: acepta `Union[Path, str]` para `base_path`
- `exam_generator.py`:
  - Mejora en `__init__()`: acepta `Union[Path, str]` para `base_path`
  - Mejora en `generate_exam_frontmatter()`: tipo para `tags` (ahora `Optional[List[str]]`)
- `exercise_analyzer.py`:
  - Mejora en `analyze()`: tipo de retorno más específico (`Dict[str, Optional[str | int | float | List[str]]`)
- `llm_providers.py`:
  - Añadido `Union` a imports para mejor flexibilidad de tipos

**Beneficios**:
- Mejor detección de errores en tiempo de desarrollo
- Mejor integración con IDEs (autocompletado y documentación)
- Mayor claridad en la interfaz de métodos
- Facilita el mantenimiento futuro y refacuración
- Mejor interoperabilidad entre componentes

## Mejoras Futuras Planeadas

- [x] Estandarización completa de type hints
- [x] Implementación de caché para respuestas de LLM con optimizaciones de memoria y I/O
- [x] Implementación de proveedores asíncronos de LLM
- [x] Centralización de imports de dependencias opcionales
- [ ] Mejora de logging con contexto estructurado
- [ ] Añadir tests unitarios para todos los proveedores LLM
- [ ] Documentar métricas internas de rendimiento
- [ ] Considerar implementación de streaming para respuestas largas

## Unificación de Política de Errores (v0.2.0)

**Objetivo**: Establecer una política consistente de manejo de errores para mejorar la mantenibilidad y facilitar el diagnóstico.

**Documentación**: Política completa en `docs/ERROR_HANDLING.md`

**Principios Implementados**:

### 1. Jerarquía de Excepciones Personalizadas

```python
EvolutiaError (base)
├── ConfigurationError
├── ProviderError
├── ValidationError
├── MaterialExtractionError
├── ExamGenerationError
└── RAGError
```

### 2. Políticas de Manejo de Errores

- **Fail-fast para configuración incorrecta**: Errores que previenen la inicialización lanzan excepciones inmediatamente
- **Graceful degradation para recursos externos**: Errores de API, archivos, etc. loguean y retornan valores seguros
- **Logging contextual**: Todos los errores incluyen contexto suficiente (nombre del componente, información relevante)
- **Diferenciación de niveles**:
  - `logger.error()`: Errores que previenen la operación principal
  - `logger.warning()`: Errores recuperables/esperados
  - `logger.info()`: Operaciones exitosas con contexto
  - `logger.debug()`: Información de diagnóstico

### 3. Mejoras por Archivo

**evolutia/llm_providers.py**:
- Logging mejorado en `_setup_openai_client()` y `_openai_generate_content()` con prefijos de componente
- Mensajes de error más específicos con información de endpoint, timeout, etc.
- Retorno consistente: `None` para errores recuperables, logging para diagnóstico

**evolutia/variation_generator.py**:
- Mejora en `_get_provider()` con logging de inicialización exitosa
- Mensajes de warning cuando provider o content son `None`
- Contexto agregado en mensajes de error (nombre del proveedor, parámetros)

**evolutia/evolutia_engine.py**:
- Logging mejorado en `initialize_rag()` con instrucciones de instalación
- Mejora en `_generate_single_variation()` con información de intentos
- Mejora en `_generate_creation_mode()` con contexto de topic, tags, complexity

**evolutia/material_extractor.py**:
- Mensajes de warning mejorados con contexto de archivo y ruta
- Mensajes de error mejorados con información del archivo que falló

**evolutia/complexity_validator.py**:
- Logging mejorado en `validate_batch()` con información de label y número de variaciones
- Resumen al final del procesamiento del lote

**evolutia/exam_generator.py**:
- Mensajes de error mejorados con contexto de output_dir y exam_number
- Mensajes de info para confirmar generación exitosa

**evolutia/exceptions.py** (Nuevo archivo):
- Excepciones personalizadas para todos los tipos de errores
- Jerarquía clara para manejo consistente

**evolutia/__init__.py**:
- Exportación de excepciones personalizadas para uso externo

### 4. Formato de Logging

```python
# Prefijo de componente para facilitar filtrado y búsqueda
logger.info("[Componente] Mensaje con contexto")
logger.error("[Componente] Error: {detalle}")

# Ejemplos implementados
logger.info("[OpenAICompatibleProvider] Cliente OpenAI inicializado")
logger.error("[VariationGenerator] Proveedor no inicializado, no se puede generar variación")
logger.warning("[MaterialExtractor] Include no encontrado en ejercicio: /path/to/file.md")
```

### 5. Beneficios

- **Mejor diagnóstico**: Mensajes de error más claros y con contexto
- **Facilita mantenimiento**: Política consistente hace más fácil agregar/modificar código
- **Mejor integración con herramientas**: Logging estructurado facilita análisis con ELK, Grafana, etc.
- **Menos tiempo de depuración**: Contexto suficiente para identificar el problema rápidamente
- **Documentación clara**: `docs/ERROR_HANDLING.md` sirve como guía para desarrolladores

**Archivos creados/modificados**: 8 archivos (1 nuevo, 7 modificados)
**Líneas de código mejoradas**: ~100 líneas de logging y manejo de errores

## Sistema de Caché LLM (v0.2.0)

**Objetivo**: Reducir costos y tiempo de ejecución almacenando respuestas de LLMs con optimizaciones de memoria y I/O.

### Módulo: `evolutia/cache/llm_cache.py`

#### Características Principales

1. **Caché en memoria con persistencia opcional**
   - Estructura en RAM para acceso rápido
   - Persistencia a disco para retención entre sesiones
   - Metadatos de caché (version, last_persisted, entries_count, hits, misses)

2. **Write-behind con debounce**
   - No escribe a disco en cada `put()` (antes sí)
   - Espera 5 segundos de inactividad antes de persistir
   - Worker thread dedicado para persistencia
   - Persistencia forzada al salir del programa (atexit)

3. **Límite de memoria RAM configurable**
   - Parámetro `max_memory_mb` (default: 500MB)
   - Tracking de tamaño por entrada con `entry_sizes`
   - Estimación de tamaño con `sys.getsizeof()`
   - Eviction automático basado en RAM

4. **LRU eviction**
   - Elimina entradas menos recientes cuando se excede `max_size`
   - Elimina entradas cuando se excede límite de memoria
   - Mantenimiento de `timestamps` para tracking de antigüedad

5. **TTL configurable**
   - Tiempo de vida en horas (0 = sin expiración)
   - Verificación automática en cada `get()`
   - Eliminación de entradas expiradas

6. **Filtrado de respuestas inválidas**
   - Rechazo de respuestas vacías
   - Rechazo de respuestas muy cortas (< 20 chars)
   - Rechazo de respuestas con indicadores de error

#### Clase Principal: `LLMCache`

**Parámetros de inicialización**:
- `max_size`: Número máximo de entradas (default: 1000)
- `ttl_hours`: Tiempo de vida en horas (default: 24)
- `persist_to_disk`: Habilita persistencia a disco (default: True)
- `cache_dir`: Directorio de caché (default: `./storage/cache/llm`)
- `debounce_seconds`: Tiempo de debounce para persistencia (default: 5.0)
- `max_memory_mb`: Límite máximo de memoria en MB (default: 500)

**Métodos principales**:
- `get(prompt, provider, model)`: Obtiene respuesta del caché
- `put(prompt, provider, model, response, metadata)`: Almacena respuesta en caché
- `clear()`: Limpia todo el caché
- `get_stats()`: Obtiene estadísticas del caché (entries, hits, misses, hit_rate, memory_mb, etc.)

**Estadísticas expandidas**:
```python
{
    'entries': 150,
    'hits': 450,
    'misses': 120,
    'hit_rate': 0.79,
    'max_size': 1000,
    'ttl_hours': 24,
    'persist_to_disk': True,
    'cache_dir': './storage/cache/llm',
    'memory_mb': 42.5,
    'memory_limit_mb': 500,
    'memory_usage_percent': 8.5
}
```

#### Integración con Proveedores LLM

Todos los proveedores LLM soportan caché mediante el parámetro `cache`:

```python
from evolutia.llm_providers import OpenAIProvider, LLMCache

# Crear caché
cache = LLMCache(max_size=1000, ttl_hours=24, max_memory_mb=500)

# Crear proveedor con caché
provider = OpenAIProvider(cache=cache)
```

El caché se activa automáticamente cuando se pasa una instancia de `LLMCache`.

#### Optimizaciones Implementadas

1. **Reducción de I/O de disco**
   - **Antes**: Cada `put()` escribía a disco inmediatamente
   - **Ahora**: Solo escribe después de 5 segundos de inactividad
   - **Impacto**: Reducción dramática de operaciones de I/O

2. **Límite de memoria**
   - **Antes**: Sin límite, podía saturar RAM con muchos caché hits
   - **Ahora**: Eviction automático al exceder límite configurable
   - **Impacto**: Uso de memoria controlado y predecible

3. **Estadísticas expandidas**
   - **Antes**: Solo entries, hits, misses, hit_rate
   - **Ahora**: Agrega memory_mb, memory_limit_mb, memory_usage_percent
   - **Impacto**: Mejor monitoreo y debugging de uso de memoria

#### Tests

- 13 tests nuevos para `LLMCache`
- Tests para: inicialización, put/get, TTL expiration, max_size LRU eviction, filtrado de respuestas, clear, stats

## Proveedores Asíncronos de LLM (v0.2.0)

**Objetivo**: Mejorar rendimiento para operaciones I/O-bound como llamadas a APIs de LLM usando async/await.

### Módulo: `evolutia/async_llm_providers.py`

#### Jerarquía de Clases

```
AsyncLLMProvider (ABC)
 ├── AsyncOpenAIProvider
 ├── AsyncAnthropicProvider
 └── AsyncGeminiProvider
```

#### Diseño

Los proveedores asíncronos actúan como wrappers sobre los proveedores síncronos existentes, ejecutando las operaciones en un thread pool executor usando `run_in_executor()`.

Esto proporciona:
- Compatibilidad con código síncrono existente
- Mejor throughput para operaciones I/O-bound
- Control de concurrencia con `asyncio.Semaphore`

#### Clases Principales

**AsyncLLMProvider** (Clase Base Abstracta):
- Constantes: `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_MAX_TOKENS`, `DEFAULT_TEMPERATURE`
- Método abstracto: `generate_content(prompt, system_prompt, **kwargs)` (async)

**AsyncOpenAIProvider**:
- Wraps `OpenAIProvider` síncrono
- Ejecuta `generate_content()` en un executor

**AsyncAnthropicProvider**:
- Wraps `AnthropicProvider` síncrono
- Ejecuta `generate_content()` en un executor

**AsyncGeminiProvider**:
- Wraps `GeminiProvider` síncrono
- Ejecuta `generate_content()` en un executor

#### Integración con EvolutiaEngine

**Nuevo método**: `generate_variations_async(selected_exercises, args, max_workers)`

```python
async def _generate_single_variation_async(
    self,
    generator: Union['VariationGenerator', 'EnhancedVariationGenerator'],
    validator: Union['ComplexityValidator', 'ConsistencyValidator'],
    exercise_base: Dict,
    analysis: Dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore
) -> Optional[Dict]:
    """Helper asíncrono para generar una única variación."""
    async with semaphore:
        # Ejecutar generación síncrona en un thread
        loop = asyncio.get_event_loop()
        variation = await loop.run_in_executor(
            None,
            generator.generate_variation,
            exercise_base,
            analysis,
            args.type
        )
        return variation
```

**Características**:
- Usa `asyncio.Semaphore` para limitar concurrencia
- Usa `asyncio.as_completed()` para procesar resultados a medida que llegan
- Mantiene compatibilidad con versión síncrona (`generate_variations_parallel()`)

#### Beneficios vs ThreadPoolExecutor

| Aspecto | ThreadPoolExecutor | Async/Await |
|--------|-------------------|-------------|
| Overhead por thread | Alto (cada worker es un thread del SO) | Bajo (coroutines son ligeros) |
| Escalabilidad | Limitado por número de threads | Miles de coroutines concurrentes |
| I/O blocking | Bloquea el thread esperando respuesta | Non-blocking, yield a event loop |
| Uso de memoria | Alto (stack por thread) | Bajo (compartido) |
| Latencia | Moderada | Más baja para I/O-bound |

#### Uso

**Activar generación asíncrona**:

```python
from evolutia.evolutia_engine import EvolutiaEngine

engine = EvolutiaEngine(base_path, config_path)

# Usar método asíncrono
variations = engine.generate_variations_async(
    selected_exercises,
    args,
    max_workers=5
)
```

El método síncrono `generate_variations_parallel()` sigue disponible para compatibilidad.

#### Notas de Implementación

- Usa `run_in_executor(None, func, *args)` para ejecutar en thread pool default
- El event loop de asyncio maneja la concurrencia
- Compatible con Python 3.7+
- No requiere cambios en proveedores síncronos existentes
- El mismo código funciona para ambos modos (sync/async)

## Imports Centralizados (v0.2.0)

**Objetivo**: Reducir duplicación de código y mejorar mantenibilidad centralizando imports de dependencias opcionales.

### Módulo: `evolutia/imports.py`

#### Clase Principal: `OptionalImports`

Gestiona imports de dependencias opcionales de forma controlada:

```python
from evolutia.imports import OptionalImports

# Importar ChromaDB
chromadb, settings = OptionalImports.get_chromadb()

# Importar sentence-transformers
SentenceTransformer = OptionalImports.get_sentence_transformers()

# Verificar disponibilidad de RAG
if OptionalImports.check_rag_available():
    # Usar funcionalidad RAG
    pass
```

#### Métodos Disponibles

- `get_chromadb()`: Importa ChromaDB y Settings
- `get_sentence_transformers()`: Importa SentenceTransformer
- `get_openai()`: Importa OpenAI
- `get_anthropic()`: Importa Anthropic
- `get_google_generativeai()`: Importa google.generativeai
- `check_rag_available()`: Verifica si todas las dependencias de RAG están disponibles
- `get_module(module_name)`: Importa cualquier módulo por nombre

#### Características

- **Caché de imports**: Solo importa cada módulo una vez
- **Logging informativo**: Mensajes de warning cuando un módulo no está disponible
- **Mensajes de error claros**: Instrucciones de instalación cuando falta una dependencia
- **Funciones de conveniencia**: Para compatibilidad con código existente

#### Beneficios

- **Menos duplicación**: No hay múltiples bloques try/except en todo el proyecto
- **Mantenibilidad centralizada**: Cambios en lógica de imports en un solo lugar
- **Consistencia**: Mismos mensajes de error y comportamiento en todo el proyecto
- **Testing más fácil**: Mock simple de una sola clase

#### Ejemplo de Uso

**Antes** (duplicado en múltiples archivos):

```python
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

if SENTENCE_TRANSFORMERS_AVAILABLE:
    # Usar sentence-transformers
    pass
```

**Ahora** (centralizado):

```python
from evolutia.imports import OptionalImports

SentenceTransformer = OptionalImports.get_sentence_transformers()

if SentenceTransformer:
    # Usar sentence-transformers
    pass
```
**Tests**: Todos los tests continúan pasando (20/20)

