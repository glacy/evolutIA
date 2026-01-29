"""
Generador de variaciones de ejercicios con mayor complejidad.
Utiliza APIs de IA para generar variaciones inteligentes.
"""
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from pathlib import Path

# Imports for new Provider system
from llm_providers import get_provider

# Cargar variables de entorno explícitamente desde el directorio del script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class VariationGenerator:
    """Genera variaciones de ejercicios con mayor complejidad."""
    
    def __init__(self, api_provider: str = "openai"):
        """
        Inicializa el generador.
        
        Args:
            api_provider: Proveedor de API ('openai', 'anthropic', 'gemini' o 'local')
        """
        self.api_provider = api_provider
        self.base_url = None # For local overrides
        self.local_model = None # For local overrides
        self.model_name = None # For overrides
        
        # Lazy initialization or on-demand to support dynamic config updates?
        # For now we'll initialize the provider on use or here if config is static.
        # But `evolutia_engine` sets attributes after init.
        # So we'll get the provider in usage methods or rebuild it.
        # Ideally `VariationGenerator` should receive a config dict.
        self._provider_instance = None
    
    def _get_provider(self):
        """Lazy loader para el proveedor, permitiendo configuración tardía de props."""
        if self._provider_instance:
            return self._provider_instance
            
        kwargs = {}
        if self.model_name:
            kwargs['model_name'] = self.model_name
        elif self.local_model and self.api_provider == 'local':
             kwargs['model_name'] = self.local_model
             
        if self.base_url and self.api_provider == 'local':
            kwargs['base_url'] = self.base_url

        try:
            self._provider_instance = get_provider(self.api_provider, **kwargs)
        except ValueError as e:
            logger.error(f"Error inicializando proveedor: {e}")
            return None
            
        return self._provider_instance

    def _create_prompt(self, exercise: Dict, analysis: Dict) -> str:
        """
        Crea el prompt para la generación de variaciones.
        """
        content = exercise.get('content', '')
        solution = exercise.get('solution', '')
        
        prompt = f"""Eres un experto en métodos matemáticos para física e ingeniería. Tu tarea es crear una variación de un ejercicio que sea MÁS COMPLEJA que el original, pero manteniendo el mismo tipo de problema y conceptos fundamentales.

EJERCICIO ORIGINAL:
{content}

SOLUCIÓN ORIGINAL (para referencia):
{solution[:1000] if solution else "No disponible"}

ANÁLISIS DEL EJERCICIO ORIGINAL:
- Tipo: {analysis.get('type', 'desconocido')}
- Pasos en solución: {analysis.get('solution_steps', 0)}
- Variables: {', '.join(analysis.get('variables', [])[:10])}
- Conceptos: {', '.join(analysis.get('concepts', []))}
- Complejidad matemática: {analysis.get('math_complexity', 0):.2f}

INSTRUCCIONES PARA LA VARIACIÓN:
1. AUMENTA la complejidad matemática de una o más de estas formas:
   - Agrega más variables independientes
   - Combina múltiples teoremas o conceptos en un solo ejercicio
   - Agrega pasos intermedios adicionales
   - Introduce condiciones especiales o casos límite
   - Modifica sistemas de coordenadas (de cartesianas a cilíndricas/esféricas, etc.)
   - Aumenta el número de dimensiones o componentes

2. MANTÉN:
   - El mismo tipo de ejercicio (demostración, cálculo, aplicación)
   - Los conceptos matemáticos fundamentales
   - El formato y estilo del ejercicio original
   - El uso de notación matemática LaTeX correcta

3. FORMATO:
   - Usa bloques de matemáticas con :::{{math}} para ecuaciones display
   - Usa $...$ para matemáticas inline
   - Mantén el español como idioma
   - Incluye contexto físico o de ingeniería si aplica

GENERA SOLO EL ENUNCIADO DEL EJERCICIO VARIADO (sin solución). El ejercicio debe ser claramente más complejo que el original."""
        
        return prompt
    
    def _create_quiz_prompt(self, context_info: Dict) -> str:
        """
        Crea el prompt para ejercicios de selección única.
        """
        content = context_info.get('content', '')
        
        prompt = f"""Eres un experto docente universitario en física y matemáticas.
Tu tarea es crear una pregunta de SELECCIÓN ÚNICA (Multiple Choice) de alta calidad y complejidad, basada en el siguiente contexto o ejercicio:

CONTEXTO/EJERCICIO BASE:
{content}

REQUISITOS:
1. Nivel: Universitario avanzado.
2. ENFOQUE: CONCEPTUAL. La pregunta debe evaluar la comprensión profunda de conceptos, teoremas, definiciones o propiedades.
   - EVITA preguntas que requieran cálculos largos o procedimentales.
   - PREFIERE preguntas sobre implicaciones teóricas, relaciones entre conceptos, o interpretaciones físicas.
   - ESTILO: Directo, conciso, tipo "completar la frase" o "seleccionar la afirmación verdadera".
   - RESTRICCIÓN IMPORTANTE: NO generes preguntas de tipo Falso/Verdadero o Sí/No. Deben ser 4 opciones conceptuales distintas.

EJEMPLO DE ESTILO DESEADO:
"El producto escalar de vectores perpendiculares es __________."
Opciones:
A) nulo
B) unitario
C) positivo
D) negativo

3. Formato: Selección única con 4 opciones (A, B, C, D).
4. Solo UNA opción debe ser correcta.
5. Las otras 3 opciones (distractores) deben ser plausibles y basadas en errores conceptuales comunes.
6. Incluye una retroalimentación/explicación detallada.

SALIDA OBLIGATORIA: JSON
Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura (sin bloques de código markdown):

{{
  "question": "Enunciado de la pregunta en LaTeX/MyST...",
  "options": {{
    "A": "Opción A...",
    "B": "Opción B...",
    "C": "Opción C...",
    "D": "Opción D..."
  }},
  "correct_option": "A",
  "explanation": "Explicación detallada..."
}}
"""
        return prompt

    def generate_variation(self, exercise: Dict, analysis: Dict, exercise_type: str = "development") -> Optional[Dict]:
        """
        Genera una variación más compleja de un ejercicio.
        """
        # 0. Get Provider
        provider = self._get_provider()
        if not provider:
            return None

        # 1. Crear prompt
        if exercise_type == 'multiple_choice':
             context_info = {
                'content': f"Ejercicio Base:\n{exercise.get('content')}\n\nSolución Base:\n{(exercise.get('solution') or '')[:500]}..."
            }
             prompt = self._create_quiz_prompt(context_info)
        else:
            prompt = self._create_prompt(exercise, analysis)
        
        # 2. Call API via Provider
        variation_content = provider.generate_content(
            prompt, 
            system_prompt="Eres un experto en métodos matemáticos para física e ingeniería. Generas ejercicios académicos de alta calidad con notación matemática LaTeX correcta."
        )
        
        if not variation_content:
            return None

        # 3. Process Result
        variation_solution = "Solución no generada en modo simple."
        
        if exercise_type == 'multiple_choice':
            try:
                import json
                clean_content = variation_content.replace('```json', '').replace('```', '').strip()
                
                # Fix common latex backslash issues simpler approach
                clean_content_fixed = clean_content.replace('\\', '\\\\').replace('\\\\"', '\\"')
                
                try:
                     data = json.loads(clean_content_fixed, strict=False)
                except json.JSONDecodeError:
                    # Last ditch effort: try valid json if original was valid
                     data = json.loads(clean_content, strict=False)

                variation_content = f"{data['question']}\n\n"
                for opt, text in data['options'].items():
                    variation_content += f"- **{opt})** {text}\n"
                
                variation_solution = f"**Respuesta Correcta: {data['correct_option']}**\n\n{data['explanation']}"
            except Exception as e:
                logger.error(f"Error parseando JSON de quiz en base variation: {e}")
                # variation_content se queda con el raw
        
        return {
            'variation_content': variation_content,
            'variation_solution': variation_solution,
            'original_frontmatter': exercise.get('frontmatter', {}),
            'original_label': exercise.get('label'),
            'type': exercise_type
        }
    
    def generate_variation_with_solution(self, exercise: Dict, analysis: Dict) -> Optional[Dict]:
        """
        Genera una variación con su solución.
        """
        # Primero generar el ejercicio
        variation = self.generate_variation(exercise, analysis)
        
        if not variation:
            return None
            
        provider = self._get_provider()
        if not provider: return None
        
        # Luego generar la solución
        solution_prompt = f"""Eres un experto en métodos matemáticos para física e ingeniería. Resuelve el siguiente ejercicio paso a paso, mostrando todos los cálculos y procedimientos.

EJERCICIO:
{variation['variation_content']}

INSTRUCCIONES:
1. Resuelve el ejercicio de forma completa y detallada
2. Muestra todos los pasos intermedios
3. Usa notación matemática LaTeX correcta
4. Explica el razonamiento cuando sea necesario
5. Usa bloques :::{{math}} para ecuaciones display y $...$ para inline
6. Escribe en español

GENERA LA SOLUCIÓN COMPLETA:"""
        
        solution_content = provider.generate_content(solution_prompt)
        
        if solution_content:
            variation['variation_solution'] = solution_content
        
        return variation

