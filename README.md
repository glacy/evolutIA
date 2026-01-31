# EvolutIA: Generador de preguntas de examen

Sistema automatizado para generar preguntas de examen desafiantes basadas en materiales didácticos existentes (lecturas, prácticas, tareas). El sistema aumenta la complejidad matemática de los ejercicios mientras mantiene el formato y estructura familiar.

## Características Principales

- **Multi-Modo**:
    - **Variación**: Incrementa la complejidad de ejercicios existentes.
    - **Creación**: Genera ejercicios nuevos desde cero basados en temas y tags.
- **RAG (Retrieval-Augmented Generation)**: Usa tus propios apuntes y ejercicios previos como contexto para generar contenido más alineado al curso.
- **Multi-Proveedor**: Soporte nativo para OpenAI (GPT-4), Anthropic (Claude 3), Google (Gemini 1.5) y Modelos Locales (via Ollama/LM Studio).
- **Análisis de Complejidad**: Valida automáticamente que las nuevas preguntas sean matemáticamente más exigentes.
- **Formato MyST**: Salida compatible con Curvenote y Jupyter Book.

---

## Instalación

### Requisitos Previos
- Python 3.8 o superior
- API Key de tu proveedor preferido (OpenAI, Anthropic, Google) o un servidor local (Ollama).

### Opción 1: Instalación desde PyPI (Recomendada)
Para uso general, instala directamente el paquete:

```bash
# Crear entorno virtual (Recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar
pip install evolutia
```

### Opción 2: Instalación desde Fuente (Desarrollo)
Si deseas modificar el código o contribuir:

```bash
git clone https://github.com/glacy/evolutIA.git
cd evolutia
pip install -e .
```

### Configuración Inicial
Crea un archivo `.env` en la raíz de tu proyecto con tus credenciales:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

---

## Guía de Uso Rápido

El comando principal es `evolutia`. Aquí tienes los casos de uso más comunes:

### 1. Generar Variaciones (Modo Clásico)
Toma ejercicios existentes de un tema y crea versiones más complejas.

```bash
# Generar 3 variaciones del tema 'analisis_vectorial'
evolutia --tema analisis_vectorial --num_ejercicios 3 --output examenes/parcial1
```

O variar ejercicios específicos por su etiqueta (Label):
```bash
evolutia --tema analisis_vectorial --label ex1-s1 ex2-s1 --output examenes/recuperacion
```

### 2. Crear Nuevos Ejercicios (Modo Creación)
Genera ejercicios desde cero sin necesitar un "ejercicio semilla".

```bash
# Crear 3 ejercicios nuevos sobre 'numeros_complejos'
evolutia --mode creation --tema numeros_complejos --num_ejercicios 3 --output examenes/quiz1
```

### 3. Usar RAG (Contexto del Curso)
Enriquece la generación indexando tus lecturas y prácticas.

```bash
# La primera vez, usa --reindex para leer tus materiales
evolutia --tema matrices --num_ejercicios 3 --use_rag --reindex --output examenes/final

# Consultas posteriores (usa el índice ya creado)
evolutia --tema matrices --num_ejercicios 3 --use_rag --output examenes/final
```

### 4. Consultar tu Base de Conocimiento
Pregúntale al sistema qué sabe sobre un concepto (útil para verificar RAG):
```bash
evolutia --query "Teorema de Stokes"
```

---

## Configuración Avanzada

EvolutIA es altamente configurable a través del archivo `evolutia_config.yaml` o argumentos CLI.

### Archivo de Configuración
Puedes colocar un `evolutia_config.yaml` en la raíz de tu carpeta de curso. Si no existe, puedes generarlo o ver el estado actual con:

```bash
# Analiza tu estructura de carpetas y genera/actualiza la config
evolutia --analyze
# O explícitamente usando el script auxiliar
python evolutia/config_manager.py
```

### Argumentos CLI Disponibles

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--tema` | Identificador del tema (carpeta) | **Requerido** |
| `--output` | Carpeta de salida | **Requerido** |
| `--num_ejercicios` | Cantidad a generar | 1 |
| `--complejidad` | Nivel objetivo (`media`, `alta`, `muy_alta`) | `alta` |
| `--api` | Proveedor (`openai`, `anthropic`, `gemini`, `local`) | `openai` |
| `--type` | Tipo de pregunta (`problem`, `multiple_choice`) | `problem` |
| `--no_generar_soluciones` | Omite la creación de archivos de solución | False |

### Uso con Modelos Locales (Offline)
Para usar modelos como Llama 3 o Mistral sin costo de API:

1. Ejecuta tu servidor (ej. `ollama serve`).
2. Configura `evolutia_config.yaml` (opcional, si usas defaults de Ollama no es necesario):
   ```yaml
   local:
     base_url: "http://localhost:11434/v1"
     model: "llama3"
   ```
3. Ejecuta con el flag local:
   ```bash
   evolutia --tema basicos --api local --output prueba_local
   ```

---

## Gestión de Materiales (Cómo "ve" los ejercicios EvolutIA)

Para que el sistema encuentre tus ejercicios y lecturas, utiliza una estrategia de descubrimiento basada en carpetas y metadatos.

1. **Escaneo de Carpetas**: Busca archivos `.md` dentro de la carpeta del tema (ej: `./analisis_vectorial/`).
2. **Tags y Metadatos**: Para archivos fuera de esa carpeta (ej. en `tareas/`), el sistema lee el *frontmatter* YAML. Incluye el tag del tema para hacerlo visible:

```yaml
---
title: Tarea 1
tags: 
  - analisis_vectorial  # <--- Este tag permite que evolutia encuentre el archivo
  - stokes
---
```

### Trazabilidad
Los ejercicios generados heredan los tags de sus "padres". El archivo final del examen (`examenX.md`) resume todos los temas cubiertos.

---

## Estructura del Proyecto

Se recomienda la siguiente estructura para tus cursos:

```
MiCurso/
├── evolutia_config.yaml      # Configuración específica del curso
├── analisis_vectorial/       # Materiales del tema 1
│   ├── lectura.md
│   └── practica.md
├── matrices/                 # Materiales del tema 2
├── examenes/                 # Salida generada por EvolutIA
└── .env                      # API Keys (no subir a git)
```

> **Nota para usuarios antiguos**: Anteriormente se recomendaba usar Git Submodules. Ese método ha sido archivado. Si lo necesitas, consulta [docs/legacy/GUIDE_SUBMODULES.md](docs/legacy/GUIDE_SUBMODULES.md).

## Contribuciones y Desarrollo

El código fuente está organizado modularmente en `evolutia/`:
- `evolutia_engine.py`: Orquestador principal.
- `variation_generator.py`: Lógica de prompts y llamadas a LLMs.
- `rag/`: Subsistema de indexación y recuperación.

Para reportar bugs o mejoras, por favor visita el repositorio en GitHub.

## Licencia
Apache 2.0
