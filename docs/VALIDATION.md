# Validación de Inputs en EvolutIA

EvolutIA incluye validación exhaustiva de argumentos CLI y configuración para prevenir errores en tiempo de ejecución y proporcionar mensajes de error claros.

## Validación de Argumentos CLI

El módulo `evolutia.validation.args_validator` proporciona validación automática de todos los argumentos de línea de comandos.

### Validaciones Implementadas

#### Argumentos Generales

| Argumento | Validación | Mensaje de Error |
|-----------|------------|------------------|
| `--complejidad` | Debe ser uno de: `media`, `alta`, `muy_alta` | `--complejidad debe ser uno de ['alta', 'media', 'muy_alta']` |
| `--api` | Debe ser un proveedor válido: `openai`, `anthropic`, `local`, `gemini`, `deepseek`, `generic` | `--api debe ser uno de [...]` |
| `--mode` | Debe ser `variation` o `creation` | `--mode debe ser uno de ['creation', 'variation']` |
| `--type` | Debe ser `development` o `multiple_choice` | `--type debe ser uno de [...]` |

#### Argumentos Numéricos

| Argumento | Validación | Mensaje de Error/Warning |
|-----------|------------|-------------------------|
| `--num_ejercicios` | Debe ser positivo | `--num_ejercicios debe ser positivo` |
| `--num_ejercicios` | Warning si > 50 | `--num_ejercicios es muy alto, esto puede generar un costo significativo en API` |
| `--workers` | Debe estar entre 1 y 20 | `--workers debe estar entre 1 y 20` |
| `--workers` | Warning si > 20 | `--workers es alto, esto puede causar rate limiting de API` |

#### Rutas y Archivos

| Argumento | Validación | Mensaje de Error |
|-----------|------------|------------------|
| `--base_path` | Debe existir | `--base_path no existe: /ruta` |
| `--base_path` | Debe ser un directorio | `--base_path no es un directorio: /ruta` |
| `--config` | Debe existir | `--config no existe: /ruta` |
| `--config` | Debe ser un archivo | `--config no es un archivo: /ruta` |
| `--output` | Parent debe ser directorio válido | `--output no es un directorio válido` |
| `--output` | Debe tener permisos de escritura | `--output no tiene permisos de escritura` |

#### Combinaciones de Argumentos

| Combinación | Validación | Mensaje de Error/Warning |
|-------------|------------|-------------------------|
| `--mode creation` | Requiere `--tema` | `Mode creation requiere --tema` |
| `--mode creation` | Warning sin `--tags` | `Mode creation sin --tags puede generar ejercicios genéricos` |
| `--api generic` | Warning sin `--base_url` | `--api generic requiere --base_url, usando valor por defecto` |
| `--use_rag` / `--query` / `--reindex` | Verifica dependencias RAG | `RAG solicitado pero faltan dependencias...` |

## Validación de Configuración

El módulo `evolutia.validation.config_validator` valida toda la configuración del sistema en `evolutia_config.yaml`.

### Estructura de Configuración

```yaml
api:
  default_provider: openai
  providers:
    openai:
      model: gpt-4
      max_tokens: 2000
      temperature: 0.7
    local:
      base_url: http://localhost:11434/v1
      model: llama3
      timeout: 300

paths:
  base_path: ..
  materials_directories: auto  # o lista de temas

exam:
  default:
    subject: "Física - II semestre 2025"
    points_per_exercise: 25
    duration_hours: 2.0
  keywords:
    analisis_vectorial: [vector, integral, gradiente]

generation:
  max_workers: 5
  request_delay: 1.0
  retry_attempts: 3
  llm_params:
    default_temperature: 0.7
    default_max_tokens: 2000
  complexty:
    min_improvement_percent: 10
    required_improvements_count: 2

rag:
  vector_store:
    type: chromadb
    persist_directory: ./storage/vector_store
  embeddings:
    provider: openai
    model: text-embedding-3-small
    batch_size: 100
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
  chunking:
    chunk_size: 1000
    chunk_overlap: 100
```

### Validaciones por Sección

#### API

| Campo | Validación | Rango/Formato |
|-------|------------|---------------|
| `default_provider` | Debe ser un proveedor válido | `openai`, `anthropic`, `local`, `gemini`, `deepseek`, `generic` |
| `providers.openai.model` | Debe ser string | - |
| `providers.openai.max_tokens` | Debe ser entero positivo | > 0 |
| `providers.openai.temperature` | Debe estar en rango | [0, 2] |
| `providers.anthropic.temperature` | Debe estar en rango | [0, 1] |
| `providers.local.base_url` | Debe ser URL válida | Debe empezar con `http://` o `https://` |
| `providers.local.timeout` | Debe ser numérico positivo | > 0 |
| `providers.generic.base_url` | Debe ser URL válida | Debe empezar con `http://` o `https://` |

#### Paths

| Campo | Validación |
|-------|------------|
| `base_path` | Debe existir y ser un directorio |
| `materials_directories` | Si es lista, verifica que los temas existan |

#### Exam

| Campo | Validación | Rango |
|-------|------------|-------|
| `default.subject` | Debe ser string | - |
| `default.points_per_exercise` | Debe ser entero positivo | > 0 |
| `default.duration_hours` | Debe estar en rango | (0, 24] |
| `keywords` | Debe ser diccionario de listas | Cada lista debe contener solo strings |

#### Generation

| Campo | Validación | Rango |
|-------|------------|-------|
| `max_workers` | Debe estar en rango | [1, 50] |
| `request_delay` | Debe ser numérico no negativo | >= 0 |
| `retry_attempts` | Debe ser entero no negativo | >= 0 |
| `llm_params.default_temperature` | Debe estar en rango | [0, 2] |
| `llm_params.default_max_tokens` | Debe ser entero positivo | > 0 |
| `complexity.min_improvement_percent` | Debe estar en rango | [0, 100] |
| `complexity.required_improvements_count` | Debe ser entero no negativo | >= 0 |

#### RAG

| Campo | Validación | Rango/Formato |
|-------|------------|---------------|
| `vector_store.type` | Debe ser tipo válido | `chromadb` |
| `embeddings.provider` | Debe ser proveedor válido | `openai`, `sentence-transformers` |
| `embeddings.batch_size` | Debe ser entero positivo | > 0 |
| `retrieval.top_k` | Debe estar en rango | [1, 100] |
| `retrieval.similarity_threshold` | Debe estar en rango | [0, 1] |
| `chunking.chunk_size` | Debe ser entero positivo | > 0 |
| `chunking.chunk_overlap` | Debe ser entero no negativo | >= 0 |
| `chunking.chunk_overlap` | Debe ser menor que chunk_size | `overlap < size` |

## Ejemplos de Validación

### Error en CLI

```bash
$ python evolutia_cli.py --tema test --output examenes --complejidad invalido

[ArgsValidator] Errores de validación en argumentos:
[ArgsValidator]   - --complejidad debe ser uno de ['alta', 'media', 'muy_alta'], obtenido: invalido

usage: evolutia_cli.py [-h] [--tema TEMA [TEMA ...]] [--output OUTPUT] ...
```

### Warning en CLI

```bash
$ python evolutia_cli.py --tema test --output examenes --num_ejercicios 100

[ArgsValidator] --num_ejercicios es muy alto (100), esto puede generar un costo significativo en API
```

### Error en Configuración

```bash
$ python evolutia_cli.py --tema test --output examenes

[ConfigValidator] Errores de validación en configuración:
[ConfigValidator]   - api.providers.openai.temperature debe estar entre 0 y 2, obtenido: 3.0
[ConfigValidator] Continuando con configuración inválida (puede causar errores)
```

### Warning en Configuración

```bash
$ python evolutia_cli.py --tema test --output examenes

[ConfigValidator] paths.materials_directories contiene tema no existente: tema_inexistente
```

## Tests de Validación

EvolutIA incluye una suite completa de tests para los validadores:

```bash
# Ejecutar tests de validación
python -m pytest tests/test_args_validator.py tests/test_config_validator.py -v
```

### Cobertura de Tests

- **ArgsValidator**: 19 tests
  - Validaciones de argumentos generales
  - Validaciones de argumentos numéricos
  - Validaciones de rutas y archivos
  - Validaciones de combinaciones de argumentos
  - Validaciones de modos específicos

- **ConfigValidator**: 21 tests
  - Validaciones de configuración API
  - Validaciones de configuración de paths
  - Validaciones de configuración de examen
  - Validaciones de configuración de generación
  - Validaciones de configuración de RAG

## Extensión y Personalización

Para agregar nuevas validaciones:

### Agregar Validación de Argumento CLI

1. Agregar el método de validación en `ArgsValidator`:
   ```python
   def _validate_mi_nuevo_argumento(self, args: argparse.Namespace):
       """Valida mi nuevo argumento."""
       value = getattr(args, 'mi_nuevo_argumento', None)
       if value and condicion_invalida:
           self.errors.append("Mensaje de error")
   ```

2. Llamar al método desde `validate_args()`:
   ```python
   def validate_args(self, args: argparse.Namespace):
       # ... validaciones existentes
       self._validate_mi_nuevo_argumento(args)
       # ...
   ```

3. Agregar test en `tests/test_args_validator.py`:
   ```python
   def test_invalid_mi_nuevo_argumento(self, validator):
       args = argparse.Namespace(mi_nuevo_argumento='invalido')
       is_valid, errors = validator.validate_args(args)
       assert not is_valid
   ```

### Agregar Validación de Configuración

1. Agregar el método de validación en `ConfigValidator`:
   ```python
   def _validate_mi_nueva_seccion(self, section: Dict[str, Any]):
       """Valida mi nueva sección de configuración."""
       value = section.get('mi_campo')
       if value and condicion_invalida:
           self.errors.append("Mensaje de error")
   ```

2. Llamar al método desde `validate_config()`:
   ```python
   def validate_config(self, config: Dict[str, Any]):
       # ... validaciones existentes
       self._validate_mi_nueva_seccion(config.get('mi_nueva_seccion', {}))
       # ...
   ```

3. Agregar test en `tests/test_config_validator.py`:
   ```python
   def test_invalid_mi_nueva_seccion(self, validator):
       config = {'mi_nueva_seccion': {'mi_campo': 'invalido'}}
       is_valid, errors = validator.validate_config(config)
       assert not is_valid
   ```

## Convenciones de Validación

### Errores vs Warnings

- **Errores**: Bloquean la ejecución del programa. Se usan para:
  - Valores inválidos (fuera de rango, tipo incorrecto)
  - Rutas inexistentes
  - Falta de argumentos obligatorios

- **Warnings**: No bloquean la ejecución pero advierten de problemas potenciales. Se usan para:
  - Valores subóptimos pero válidos (muy alto, muy bajo)
  - Valores ausentes con defaults razonables
  - Dependencias opcionales faltantes

### Formato de Mensajes

- **Errores**: `--argumento descripción del problema, obtenido: valor`
- **Warnings**: `--argumento descripción del problema`
- **Configuración**: `sección.campo descripción del problema, obtenido: valor`

### Tipo de Retorno

```python
# ArgsValidator
def validate_args(self, args: argparse.Namespace) -> Tuple[bool, List[str]]:
    """
    Returns:
        Tupla (is_valid, error_messages)
        - is_valid: True si todos los argumentos son válidos
        - error_messages: Lista de mensajes de error (vacía si is_valid es True)
    """

# ConfigValidator
def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Returns:
        Tupla (is_valid, error_messages)
        - is_valid: True si la configuración es válida
        - error_messages: Lista de mensajes de error (vacía si is_valid es True)
    """
```
