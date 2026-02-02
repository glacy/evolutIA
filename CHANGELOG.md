# Changelog

Todas las variaciones notables de este proyecto serán documentadas en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Mantenimiento
- Migración del paquete `google.generativeai` a `google.genai` para resolver advertencias de deprecación (FutureWarning) y asegurar compatibilidad futura.

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
