# Guía para Agentes de Código (AGENTS.md)

Este documento proporciona instrucciones para agentes de IA que trabajan en el repositorio de EvolutIA.

## Comandos de Build, Lint y Test

```bash
# Instalación
pip install -e ".[rag]" pytest pytest-cov black ruff mypy

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_args_validator.py::TestArgsValidator::test_valid_basic_args -v
python -m pytest tests/ --cov=evolutia --cov-report=html

# Linting y formateo
black evolutia/ tests/
ruff check evolutia/ tests/ && ruff check --fix evolutia/ tests/
mypy evolutia/

# Todo junto
ruff check evolutia/ tests/ && black --check evolutia/ tests/ && python -m pytest tests/ -v
```

## Guía de Estilo de Código

### Importaciones
Orden: stdlib → third-party → local (grupos separados por línea en blanco)

### Type Hints
```python
from typing import Dict, List, Optional, Tuple, Union

def validate(args: argparse.Namespace) -> Tuple[bool, List[str]]:
    pass

def get(self, prompt: str) -> Optional[str]:
    pass
```

### Convenciones
- **Clases**: PascalCase (`ConfigValidator`, `LLMCache`)
- **Funciones/Métodos**: snake_case (`validate_config`, `get_cache_key`)
- **Constantes**: UPPER_SNAKE_CASE (`VALID_API_PROVIDERS`)
- **Privados**: `_leading_underscore`

### Manejo de Errores
```python
from evolutia.exceptions import EvolutiaError, ConfigurationError

# Fail-fast para config
logger.error("[Component] Config error: {e}")
raise ConfigurationError("API config requerida")

# Graceful degradation para recursos externos
try:
    return api_call()
except Exception as e:
    logger.warning(f"[Component] Error: {e}")
    return None
```

### Docstrings (estilo Google)
```python
def validate_config(self, config: Dict) -> Tuple[bool, List[str]]:
    """
    Valida configuración.

    Args:
        config: Diccionario de configuración.

    Returns:
        Tupla (is_valid, error_messages).

    Example:
        >>> is_valid, errors = validator.validate_config(config)
    """
```

### Tests
- pytest con fixtures, test classes: `TestClassName`
- Nombres: `test_descriptive_name`

## Módulos Recién Creados

### Validación (`evolutia/validation/`)
`ArgsValidator`, `ConfigValidator` - Validación de args CLI y config

### Caché (`evolutia/cache/`)
`LLMCache`, `ExerciseAnalysisCache` - Caché de respuestas LLM y análisis de ejercicios
- Features: write-behind con debounce, límite de memoria RAM, TTL, LRU eviction

### Retry (`evolutia/`)
`retry_utils` - Utilidades para manejo de errores y reintentos
- Decoradores `@retry_async` y `@retry_sync` con backoff exponencial
- Clase `CircuitBreaker` para evitar llamadas a servicios fallidos

### Imports Centralizados (`evolutia/`)
`OptionalImports` - Gestor de imports condicionales para dependencias opcionales
- Centraliza imports de ChromaDB, sentence-transformers, OpenAI, Anthropic, Gemini
- Reduce duplicación de código try/except

### Async LLM Providers (`evolutia/`)
`AsyncLLMProvider`, `AsyncOpenAIProvider`, `AsyncAnthropicProvider`, `AsyncGeminiProvider` - Proveedores asíncronos para llamadas LLM
- Más eficiente que ThreadPoolExecutor para operaciones I/O-bound
- Compatible con versión síncrona existente
- Incluye retry automático con backoff exponencial

## Git Workflow

### Branch Naming
`feature/desc` | `fix/desc` | `refactor/desc` | `docs/desc` | `test/desc`

### Commit Messages (Conventional)
`<tipo>(<alcance>): <descripción>` - Tipos: feat, fix, refactor, docs, test, chore, perf, style

### Pull Request Guidelines
1. PR de feature/fix → main
2. Descripción: resumen, tests, cambios breaking
3. Tests pasando: `python -m pytest tests/ -v`
4. Linting: `ruff check --fix evolutia/`
5. Actualizar CHANGELOG.md

## Configuración de Herramientas

### Ruff (linter, .ruff.toml)
```toml
[tool.ruff]
line-length = 100
target-version = "py38"
[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP"]
ignore = ["E501", "N806"]
```

### Black (formateador, .black.toml)
```toml
[tool.black]
line-length = 100
target-version = ['py38']
```

### MyPy (type checker, mypy.ini)
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
disallow_untyped_defs = false
check_untyped_defs = true
```

## Comandos Especiales

```bash
# Auto-descubrimiento de config
python evolutia_cli.py --analyze

# Generación de exámenes
evolutia --tema analisis_vectorial --num_ejercicios 4 --output examenes/examen1
evolutia --tema analisis_vectorial --mode creation --tags vector --output examenes/examen2
evolutia --tema analisis_vectorial --use_rag --num_ejercicios 4 --output examenes/examen3

# Debugging
LOG_LEVEL=DEBUG python evolutia_cli.py --tema test --output examenes/examen1
```

## Documentación

- `docs/ARCHITECTURE.md` - Arquitectura interna
- `docs/ERROR_HANDLING.md` - Política de errores
- `docs/VALIDATION.md` - Guía de validación
- `CHANGELOG.md` - Historial de cambios

Al agregar funcionalidad: actualizar docstrings, agregar tests, actualizar CHANGELOG.md, actualizar README.md si es API pública.
