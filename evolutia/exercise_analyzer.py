"""
Analizador de complejidad de ejercicios.
Identifica tipo, pasos, conceptos y variables de ejercicios.
"""
import re
import logging
from typing import Dict, List, Set, Optional, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from evolutia.cache.exercise_cache import ExerciseAnalysisCache

logger = logging.getLogger(__name__)

try:
    from utils.math_extractor import (
        extract_math_expressions,
        extract_variables,
        count_math_operations,
        estimate_complexity
    )
except ImportError:
    from .utils.math_extractor import (
        extract_math_expressions,
        extract_variables,
        count_math_operations,
        estimate_complexity
    )


class ExerciseAnalyzer:
    """Analiza la complejidad y estructura de ejercicios."""

    TYPE_PATTERNS = {
        'demostracion': re.compile(r'(?i)(demuestre|pruebe|verifique|muestre|justifique|demostraci[oó]n)'),
        'calculo': re.compile(r'(?i)(calcule|halle|encuentre|resuelva|eval[uú]e|calcular|obtenga|determinar)'),
        'aplicacion': re.compile(r'(?i)(aplicaci[oó]n|problema|vida real|modelo|f[íi]sic[ao]|ingenier[íi]a|econom[íi]a|contexto)')
    }

    STEP_KEYWORDS_PATTERN = re.compile(
        r'(?i)(primero|luego|despu[ée]s|finalmente|entonces|por lo tanto|conclusi[oó]n|paso|seguidamente)',
        re.MULTILINE
    )

    NUMBERED_STEPS_PATTERN = re.compile(r'^\s*\d+[\.\)]\s+', re.MULTILINE)
    EQUATION_BLOCKS_PATTERN = re.compile(r'\\begin\{(align|equation|aligned|eqnarray)\}')
    SEPARATORS_PATTERN = re.compile(r'\n\n+')

    CONCEPT_PATTERNS = {
        'integrals': [re.compile(p, re.IGNORECASE) for p in [r'(?i)integral', r'\\int', r'\\iint', r'\\iiint', r'\\oint']],
        'derivatives': [re.compile(p, re.IGNORECASE) for p in [r'(?i)derivada', r'\\frac{d}{d', r'\\[dp]artial', r'\'']],
        'limits': [re.compile(p, re.IGNORECASE) for p in [r'(?i)l[íi]mite', r'\\lim']],
        'series': [re.compile(p, re.IGNORECASE) for p in [r'(?i)serie', r'(?i)sucesi[oó]n', r'\\sum', r'convergencia']],
        'vectors': [re.compile(p, re.IGNORECASE) for p in [r'(?i)vector', r'\\vec', r'\\mathbf', r'producto punto', r'producto cruz']],
        'matrices': [re.compile(p, re.IGNORECASE) for p in [r'(?i)matriz', r'(?i)determinante', r'\\begin{pmatrix}', r'\\begin{bmatrix}', r'autovalor']],
        'coordinate_systems': [re.compile(p, re.IGNORECASE) for p in [r'(?i)coordenadas', r'(?i)polares', r'(?i)esf[ée]ricas', r'(?i)cil[íi]ndricas', r'jacobian[oa]']],
        'vector_operations': [re.compile(p, re.IGNORECASE) for p in [r'(?i)gradiente', r'(?i)divergencia', r'(?i)rotacional', r'\\nabla', r'teorema de stokes', r'teorema de green', r'teorema de la divergencia']]
    }

    def __init__(self, cache: Optional['ExerciseAnalysisCache'] = None):
        """
        Inicializa el analizador.

        Args:
            cache: Instancia opcional de ExerciseAnalysisCache para cachear análisis
        """
        self.cache = cache

    def identify_exercise_type(self, content: str) -> str:
        """
        Identifica el tipo de ejercicio.

        Args:
            content: Contenido del ejercicio

        Returns:
            Tipo de ejercicio: 'demostracion', 'calculo', 'aplicacion', 'mixto'
        """
        # Búsqueda optimizada con evaluación perezosa (short-circuit)
        # Verificamos demostración primero ya que es determinante para 'mixto'
        if self.TYPE_PATTERNS['demostracion'].search(content):
            # Si es demostración, buscamos otros tipos para ver si es mixto
            # Basta con encontrar uno de los dos para que sea mixto
            if (self.TYPE_PATTERNS['calculo'].search(content) or
                self.TYPE_PATTERNS['aplicacion'].search(content)):
                return 'mixto'
            return 'demostracion'

        # Si no es demostración, buscamos cálculo
        if self.TYPE_PATTERNS['calculo'].search(content):
            return 'calculo'

        # Finalmente aplicación
        if self.TYPE_PATTERNS['aplicacion'].search(content):
            return 'aplicacion'

        return 'calculo'  # Por defecto

    def count_solution_steps(self, solution_content: str) -> int:
        """
        Cuenta el número de pasos en una solución.

        Busca indicadores de pasos como:
        - Numeración (1., 2., etc.)
        - Palabras clave (Primero, Luego, Finalmente, etc.)
        - Bloques de ecuaciones separados

        Args:
            solution_content: Contenido de la solución

        Returns:
            Número estimado de pasos
        """
        if not solution_content:
            return 0

        # Contar numeración explícita
        numbered_steps = len(self.NUMBERED_STEPS_PATTERN.findall(solution_content))

        # Contar palabras clave de pasos
        keyword_steps = len(self.STEP_KEYWORDS_PATTERN.findall(solution_content))

        # Contar bloques de ecuaciones (align, equation)
        equation_blocks = len(self.EQUATION_BLOCKS_PATTERN.findall(solution_content))

        # Estimar pasos basado en separadores
        separators = len(self.SEPARATORS_PATTERN.findall(solution_content))

        # Tomar el máximo de los métodos
        estimated_steps = max(
            numbered_steps,
            keyword_steps // 2,  # Dividir porque pueden repetirse
            equation_blocks,
            separators // 2
        )

        return max(1, estimated_steps)  # Mínimo 1 paso

    def identify_concepts(self, content: str) -> Set[str]:
        """
        Identifica conceptos matemáticos presentes en el contenido.

        Args:
            content: Contenido a analizar

        Returns:
            Conjunto de conceptos identificados
        """
        concepts = set()

        for concept_name, patterns in self.CONCEPT_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(content):
                    concepts.add(concept_name)
                    break

        return concepts

    def analyze(self, exercise: Dict[str, Optional[str]]) -> Dict[str, Optional[str | int | float | List[str]]]:
        """
        Analiza un ejercicio completo y retorna metadatos de complejidad.

        Args:
            exercise: Diccionario con información del ejercicio
                - 'content': Contenido del ejercicio
                - 'solution': Contenido de la solución (opcional)

        Returns:
            Diccionario con análisis de complejidad
        """
        content = exercise.get('content', '')
        solution = exercise.get('solution', '')

        # Intentar caché primero
        if self.cache:
            cached_analysis = self.cache.get(exercise)
            if cached_analysis:
                logger.info(f"[ExerciseAnalyzer] Análisis obtenido del caché para exercise={exercise.get('label', 'unknown')}")
                return cached_analysis['analysis']

        # Análisis normal (cache miss)
        if not content:
            return {}

        # Extraer expresiones matemáticas
        math_expressions = extract_math_expressions(content)
        if solution:
            math_expressions.extend(extract_math_expressions(solution))

        # Extraer variables
        variables = extract_variables(math_expressions)

        # Identificar tipo
        exercise_type = self.identify_exercise_type(content)

        # Contar pasos en solución
        solution_steps = self.count_solution_steps(solution) if solution else 0

        # Identificar conceptos
        all_content = content + '\n' + (solution or '')
        concepts = self.identify_concepts(all_content)

        # Calcular complejidad matemática
        math_complexity = estimate_complexity(math_expressions)

        # Contar operaciones
        total_operations = {
            'integrals': 0,
            'derivatives': 0,
            'sums': 0,
            'vectors': 0,
            'matrices': 0,
            'functions': 0
        }
        for expr in math_expressions:
            ops = count_math_operations(expr)
            for key in total_operations:
                total_operations[key] += ops[key]

        # Calcular complejidad total
        total_complexity = (
            math_complexity +
            solution_steps * 2.0 +
            len(variables) * 0.5 +
            len(concepts) * 1.5 +
            sum(total_operations.values()) * 0.5
        )

        return {
            'type': exercise_type,
            'solution_steps': solution_steps,
            'variables': list(variables),
            'num_variables': len(variables),
            'concepts': list(concepts),
            'num_concepts': len(concepts),
            'math_complexity': math_complexity,
            'operations': total_operations,
            'total_complexity': total_complexity,
            'num_math_expressions': len(math_expressions)
        }
