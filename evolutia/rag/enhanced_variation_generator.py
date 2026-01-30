"""
Enhanced Variation Generator: Genera variaciones usando RAG.
"""
import logging
import os
from typing import Dict, Optional
import google.generativeai as genai

try:
    from ..variation_generator import VariationGenerator
except ImportError:
    # Fallback for standalone execution tests (though discouraged in package)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from variation_generator import VariationGenerator

try:
    from .rag_retriever import RAGRetriever
    from .context_enricher import ContextEnricher
except ImportError:
    from rag_retriever import RAGRetriever
    from context_enricher import ContextEnricher

logger = logging.getLogger(__name__)


class EnhancedVariationGenerator(VariationGenerator):
    """Genera variaciones usando RAG para enriquecer el contexto."""

    def __init__(self, api_provider: str = "openai", retriever: RAGRetriever = None,
                 context_enricher: ContextEnricher = None):
        """
        Inicializa el generador mejorado.

        Args:
            api_provider: Proveedor de API ('openai' o 'anthropic')
            retriever: Instancia de RAGRetriever
            context_enricher: Instancia de ContextEnricher
        """
        super().__init__(api_provider)
        self.retriever = retriever
        self.context_enricher = context_enricher or ContextEnricher()

        # Configurar Gemini si es necesario
        if self.api_provider == 'gemini':
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY no encontrada en variables de entorno")
            else:
                genai.configure(api_key=api_key)

    def _retrieve_context(self, exercise: Dict, analysis: Dict) -> Dict:
        """
        Recupera contexto relevante usando RAG.

        Args:
            exercise: Información del ejercicio original
            analysis: Análisis de complejidad

        Returns:
            Diccionario con contexto recuperado
        """
        if not self.retriever:
            return {}

        context = {}

        try:
            # Buscar ejercicios similares
            exercise_content = exercise.get('content', '')
            similar = self.retriever.retrieve_similar_exercises(
                exercise_content,
                exclude_label=exercise.get('label'),
                top_k=5
            )
            context['similar_exercises'] = similar

            # Buscar conceptos relacionados
            concepts = analysis.get('concepts', [])
            if concepts:
                related = self.retriever.retrieve_related_concepts(concepts, top_k=3)
                context['related_concepts'] = related

            # Buscar contexto de lecturas
            topic = exercise.get('source_file', {}).name if hasattr(exercise.get('source_file'), 'name') else ''
            if topic:
                reading_context = self.retriever.retrieve_reading_context(topic, top_k=2)
                context['reading_context'] = reading_context

            # Buscar ejercicios con complejidad similar (para referencia)
            target_complexity = analysis.get('total_complexity', 0)
            if target_complexity > 0:
                complexity_examples = self.retriever.retrieve_by_complexity(
                    target_complexity,
                    tolerance=0.3,
                    top_k=3
                )
                context['complexity_examples'] = complexity_examples

        except Exception as e:
            logger.warning(f"Error recuperando contexto RAG: {e}")
            context = {}

        return context

    def _create_prompt(self, exercise: Dict, analysis: Dict, context: Dict = None) -> str:
        """
        Crea el prompt enriquecido con contexto RAG.

        Args:
            exercise: Información del ejercicio original
            analysis: Análisis de complejidad del ejercicio
            context: Contexto RAG opcional (para evitar re-búsqueda)

        Returns:
            Prompt enriquecido
        """
        # Crear prompt base usando el método del padre
        base_prompt = super()._create_prompt(exercise, analysis)

        # Si no hay retriever, usar prompt base
        if not self.retriever:
            return base_prompt

        # Recuperar contexto si no se proporciona
        if context is None:
            context = self._retrieve_context(exercise, analysis)

        # Enriquecer prompt con contexto
        enriched_prompt = self.context_enricher.create_enriched_prompt(
            base_prompt,
            exercise,
            analysis,
            context
        )

        return enriched_prompt

    def generate_variation(self, exercise: Dict, analysis: Dict, exercise_type: str = "development") -> Optional[Dict]:
        """
        Genera una variación de un ejercicio existente.
        Permite generar variaciones de desarrollo o convertir a quiz conceptual.
        """
        # 1. Recuperar contexto RAG si aplica
        context = self._retrieve_context(exercise, analysis)

        # 2. Construir prompt según tipo
        if exercise_type == 'multiple_choice':
            # Enriquecer contexto para string
            context_str = self.context_enricher.format_context_dict(context)

            # Para quiz, usamos el contenido del ejercicio como base
            context_info = {
                'content': f"Ejercicio Base:\n{exercise.get('content')}\n\nSolución Base:\n{(exercise.get('solution') or '')[:500]}...\n\nContexto Adicional:\n{context_str}"
            }
            prompt = self._create_quiz_prompt(context_info)
        else:
            # Flujo normal de variación desarrollo (llamando a lógica padre modificada o directa)
            # Pasamos el contexto ya recuperado a _create_prompt
            prompt = self._create_prompt(exercise, analysis, context=context)

        # 3. Get Provider
        provider = self._get_provider()
        if not provider: return None

        # 4. Generar variación
        content = provider.generate_content(prompt, system_prompt="Eres un experto en métodos matemáticos para física e ingeniería.")

        if not content:
            return None

        # 5. Parsear respuesta
        variation_content = ""
        variation_solution = ""

        if exercise_type == 'multiple_choice':
            data = extract_and_parse_json(content)
            
            if data and 'question' in data and 'options' in data:
                variation_content = f"{data['question']}\n\n"
                for opt, text in data['options'].items():
                    variation_content += f"- **{opt})** {text}\n"
                
                variation_solution = f"**Respuesta Correcta: {data.get('correct_option', '?')}**\n\n{data.get('explanation', '')}"
            else:
                 logger.warning("No se pudo parsear el JSON del quiz (enhanced), usando contenido raw")
                 variation_content = content
        else:
            variation_content = content
            variation_solution = "Solución pendiente..."
            
            # Intento de mejora de parsing standard si el modelo siguio instrucciones
            parts = content.split("SOLUCIÓN REQUERIDA:")
            if len(parts) == 2:
                 # Si el modelo siguió las instrucciones de separar con esa marca (no siempre garantizado en simple variation)
                 pass

        variation = {
            'variation_content': variation_content,
            'variation_solution': variation_solution,
            'original_frontmatter': exercise.get('frontmatter', {}),
            'original_label': exercise.get('label'),
            'type': exercise_type
        }

        if self.retriever and context:
             variation['rag_context'] = {
                'similar_exercises_count': len(context.get('similar_exercises', [])),
                'related_concepts_count': len(context.get('related_concepts', [])),
                'reading_context_count': len(context.get('reading_context', []))
            }
             # Extraer references de similar_exercises y reading_context
             refs = []
             for ex in context.get('similar_exercises', []):
                 # Prefer label from metadata, fallback to id
                 ref_label = ex.get('metadata', {}).get('label') or ex.get('id')
                 if ref_label: refs.append(ref_label)

             for reading in context.get('reading_context', []):
                  # Reading may not have label, use id or source
                  ref_src = reading.get('metadata', {}).get('source') or reading.get('id')
                  if ref_src: refs.append(ref_src)

             if refs:
                 variation['rag_references'] = refs

        return variation

    def generate_variation_with_solution(self, exercise: Dict, analysis: Dict) -> Optional[Dict]:
        """
        Genera una variación con su solución usando RAG.
        """
        # Generar variación (ya usa RAG)
        variation = self.generate_variation(exercise, analysis)

        if not variation:
            return None
            
        provider = self._get_provider()
        if not provider: return None

        # Generar solución (usar método del padre)
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

    def generate_new_exercise_from_topic(self, topic: str, tags: list = None, difficulty: str = "alta", exercise_type: str = "development") -> Optional[Dict]:
        """
        Genera un ejercicio nuevo desde cero basado en un tema y tags.
        """
        if not self.retriever:
            logger.info("Generando sin contexto RAG (retriever no disponible)")
            # Continuar sin contexto

        tags = tags or []
        context = {}

        # Normalizar topic para manejar lista o string
        if isinstance(topic, list):
            topic_str = ", ".join(topic)
            topic_list = topic
        else:
            topic_str = topic
            topic_list = [topic]

        if self.retriever:
            # 1. Recuperar contexto teórico
            reading_context = self.retriever.retrieve_reading_context(topic_str, top_k=3)
            context['reading_context'] = reading_context

            # 2. Recuperar ejercicios relacionados para estilo
            search_terms = tags + topic_list
            related_exercises = self.retriever.retrieve_related_concepts(search_terms, top_k=3)
            context['related_exercises'] = related_exercises

        # 3. Construir prompt
        if exercise_type == 'multiple_choice':
            # Preparar info para el prompt de quiz
            context_info = {
                'content': f"Tema: {topic_str}\nTags: {', '.join(tags)}\nDificultad: {difficulty}\nContexto: {str(context)}"
            }
            prompt = self._create_quiz_prompt(context_info)
        else:
            prompt = self._create_new_exercise_prompt(topic_str, tags, context, difficulty) # Use topic_str

        # 4. Get Provider and Generate
        provider = self._get_provider()
        if not provider: return None
        
        content = provider.generate_content(prompt)

        if not content:
            return None

        # 5. Parsear respuesta
        exercise_text = ""
        solution_text = ""

        if exercise_type == 'multiple_choice':
            data = extract_and_parse_json(content)
            
            if data and 'question' in data:
                exercise_text = f"{data['question']}\n\n"
                for opt, text in data.get('options', {}).items():
                    exercise_text += f"- **{opt})** {text}\n"
                solution_text = f"**Respuesta Correcta: {data.get('correct_option', '?')}**\n\n{data.get('explanation', '')}"
            else:
                 logger.error("No se pudo parsear JSON de quiz nuevo")
                 exercise_text = content
                 solution_text = "Verificar formato generado."
        else:
            # Parseo normal de ejercicio de desarrollo
            parts = content.split("SOLUCIÓN REQUERIDA:")
            if len(parts) == 2:
                exercise_text = parts[0].replace("EJERCICIO NUEVO:", "").strip()
                solution_text = parts[1].strip()
            else:
                exercise_text = content
                solution_text = ""

        return {
            'variation_content': exercise_text,
            'variation_solution': solution_text,
            'original_frontmatter': {
                'subject': topic_str,
                'tags': tags,
                'complexity': difficulty,
                'type': exercise_type
            }
        }
