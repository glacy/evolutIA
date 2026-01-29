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
from utils.json_parser import extract_and_parse_json

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

