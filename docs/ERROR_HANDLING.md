# Política de Errores de EvolutIA

Este documento define la política unificada de manejo de errores para el proyecto EvolutIA.

## Principios Generales

1. **Fail-fast para configuración incorrecta**: Errores que previenen la inicialización deben lanzar excepciones inmediatamente
2. **Graceful degradation para recursos externos**: Errores de API, archivos, etc. deben loguear y retornar valores seguros
3. **Logging contextual**: Todos los errores deben incluir contexto suficiente para diagnóstico
4. **Diferenciación de niveles**:
   - `logger.error()`: Para errores que previenen la operación principal
   - `logger.warning()`: Para errores que se recuperan o son esperados
   - `logger.debug()`: Para información de diagnóstico

## Tipos de Errores

### 1. Errores de Configuración (Fail-fast)

**Cuándo aplicar**: Errores que previenen la inicialización correcta del sistema

**Patrón**:
```python
logger.error("API key no encontrada para PROVIDER: configure XYZ_API_KEY en .env")
return False  # o lanzar excepción si es crítico
```

**Casos**:
- API keys faltantes para proveedor activo
- Dependencias obligatorias no instaladas
- Configuración inválida que previene el inicio

### 2. Errores de API (Graceful degradation)

**Cuándo aplicar**: Errores en llamadas a APIs externas (LLMs, etc.)

**Patrón**:
```python
try:
    response = api_call()
    return response
except APIError as e:
    logger.error(f"Error llamando a {provider_name} API: {e}")
    return None
```

**Casos**:
- Fallo de llamada a API de LLM
- Timeout de API
- Error de autenticación de API
- Límite de cuota alcanzado

### 3. Errores de Archivos I/O (Graceful degradation)

**Cuándo aplicar**: Errores al leer/escribir archivos

**Patrón**:
```python
try:
    content = file.read_text()
    return content
except FileNotFoundError:
    logger.warning(f"Archivo no encontrado: {file_path}")
    return default_value
except Exception as e:
    logger.error(f"Error leyendo archivo {file_path}: {e}")
    return default_value
```

**Casos**:
- Archivo no encontrado
- Permiso denegado
- Error de lectura/escritura

### 4. Errores de Validación (Contextual)

**Cuándo aplicar**: Errores al validar datos o configuración

**Patrón**:
```python
# Para errores esperados/no críticos
if not valid:
    logger.warning(f"Validación fallida: {reason}")
    return False

# Para errores críticos
if not valid:
    raise ValueError(f"Configuración inválida: {reason}")
```

**Casos**:
- Validación de esquema de configuración
- Validación de complejidad de ejercicios
- Validación de argumentos de línea de comandos

### 5. Errores Internos (Fail-fast)

**Cuándo aplicar**: Errores inesperados que indican bugs o condiciones inválidas

**Patrón**:
```python
try:
    operation()
except Exception as e:
    logger.error(f"Error inesperado en operación: {e}")
    raise  # Re-lanzar para fail-fast
```

**Casos**:
- Bug en código interno
- Estado inconsistente
- Condición inesperada

## Retornos de Métodos

### Métodos Booleanos

```python
def initialize_rag(self) -> bool:
    """
    Returns:
        True si la inicialización fue exitosa
        False si hubo un error esperado
    """
    if not RAG_AVAILABLE:
        logger.error("RAG no disponible. Instala dependencias.")
        return False
    # ...
```

### Métodos con Retorno Opcional

```python
def generate_content(self, prompt: str) -> Optional[str]:
    """
    Returns:
        Contenido generado si la llamada fue exitosa
        None si hubo un error de API o configuración
    """
    if not self.client:
        logger.error("Cliente no inicializado")
        return None
    # ...
```

### Métodos con Retorno Complejo

```python
def validate(self, exercise: Dict) -> Dict[str, Any]:
    """
    Returns:
        Diccionario con resultado de validación:
        - 'is_valid': bool
        - 'improvements': list
        - 'warnings': list
        - 'error': Optional[str] (solo si hubo error crítico)
    """
    try:
        # validación
        return {'is_valid': True, ...}
    except Exception as e:
        return {
            'is_valid': False,
            'error': str(e),
            'warnings': [f"Error de validación: {e}"]
        }
```

## Logging Guidelines

### Formato de Mensajes

```python
logger.error(f"[Componente] Descripción del error: {contexto}")
logger.warning(f"[Componente] Advertencia: {contexto}")
logger.debug(f"[Componente] Información de diagnóstico: {contexto}")
```

### Incluir Contexto

Siempre incluir información relevante:
- Nombre del componente/módulo
- Identificador del recurso (ID, nombre de archivo, etc.)
- Mensaje de la excepción original
- Información de estado relevante

### Niveles de Logging

| Nivel | Uso | Ejemplo |
|--------|-----|---------|
| `DEBUG` | Información detallada para diagnóstico | "Intentando conectar a API... endpoint={url}" |
| `INFO` | Información normal del flujo | "Indexando 50 materiales..." |
| `WARNING` | Errores recuperables/esperados | "API key no encontrada, usando defaults" |
| `ERROR` | Errores que previenen la operación | "Error llamando a OpenAI API: timeout" |
| `CRITICAL` | Errores que requieren intervención inmediata | "Base de datos corrupta" |

## Excepciones Personalizadas

Definir excepciones específicas para casos de error claros:

```python
class EvolutiaError(Exception):
    """Excepción base para errores de Evolutia."""
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
```

## Ejemplos de Implementación

### Ejemplo 1: Inicialización de Proveedor LLM

```python
def _setup_client(self):
    """Configura el cliente de la API."""
    try:
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Cliente OpenAI inicializado correctamente")
    except ImportError:
        logger.error("Biblioteca openai no instalada. Instala con: pip install openai")
        self.client = None
    except Exception as e:
        logger.error(f"Error inesperado inicializando cliente OpenAI: {e}")
        self.client = None
```

### Ejemplo 2: Generación de Contenido

```python
def generate_content(self, prompt: str, **kwargs) -> Optional[str]:
    """Genera contenido a partir de un prompt."""
    if not self.client:
        logger.error("Cliente no inicializado, no se puede generar contenido")
        return None

    try:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000)
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"Contenido generado exitosamente (longitud: {len(content)})")
        return content
    except TimeoutError as e:
        logger.error(f"Timeout en API del proveedor: {e}")
        return None
    except APIError as e:
        logger.error(f"Error de API del proveedor: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado generando contenido: {e}")
        raise  # Fail-fast para errores inesperados
```

### Ejemplo 3: Lectura de Archivos

```python
def extract_from_file(self, file_path: Path) -> Dict:
    """Extrae ejercicios y soluciones de un archivo Markdown."""
    if not file_path.exists():
        logger.warning(f"Archivo no encontrado: {file_path}")
        return {
            'file_path': file_path,
            'frontmatter': {},
            'exercises': [],
            'solutions': []
        }

    try:
        content = read_markdown_file(file_path)
        frontmatter, content_body = extract_frontmatter(content)
        exercises = extract_exercise_blocks(content_body)
        solutions = extract_solution_blocks(content_body)

        logger.debug(f"Extrayendo {len(exercises)} ejercicios de {file_path}")
        return {
            'file_path': file_path,
            'frontmatter': frontmatter,
            'exercises': exercises,
            'solutions': solutions,
            'content_body': content_body
        }
    except Exception as e:
        logger.error(f"Error extrayendo de {file_path}: {e}")
        return {
            'file_path': file_path,
            'frontmatter': {},
            'exercises': [],
            'solutions': []
        }
```

## Checklist para Implementación

Al agregar o modificar código de manejo de errores:

- [ ] ¿El nivel de logging es apropiado (error/warning/debug)?
- [ ] ¿El mensaje de error incluye contexto suficiente?
- [ ] ¿El comportamiento (return/raise) es consistente con la política?
- [ ] ¿Se está retornando un valor seguro en caso de error?
- [ ] ¿Se está propagando excepciones críticas (fail-fast)?
- [ ] ¿El código maneja casos específicos (timeout, no encontrado, etc.)?
- [ ] ¿Hay tests que cubran los casos de error?
