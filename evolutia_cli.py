#!/usr/bin/env python3
"""
Script principal para generar exámenes a partir de materiales didácticos.
CLI Wrapper para EvolutiaEngine.
"""
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde el directorio actual (donde el usuario ejecuta el comando)
load_dotenv()

# Add current directory to path so we can import 'evolutia' package if running locally without install
sys.path.insert(0, str(Path(__file__).parent))

from evolutia.evolutia_engine import EvolutiaEngine
from evolutia.config_manager import ConfigManager
from evolutia.validation.args_validator import ArgsValidator, ValidationError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal (CLI Entry Point)."""
    parser = argparse.ArgumentParser(
        description='Genera preguntas de examen basadas en materiales didácticos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Generar examen de análisis vectorial con 4 ejercicios
  python evolutia.py --tema analisis_vectorial --num_ejercicios 4 --output examenes/examen3
  
  # Generar examen con complejidad alta usando Claude
  python evolutia.py --tema matrices --num_ejercicios 3 --complejidad alta --api anthropic
        """
    )
    
    parser.add_argument('--tema', type=str, nargs='+', help='Temas del examen')
    parser.add_argument('--num_ejercicios', type=int, default=1, help='Número de ejercicios a generar')
    parser.add_argument('--output', type=str, help='Directorio de salida')
    parser.add_argument('--complejidad', type=str, choices=['media', 'alta', 'muy_alta'], default='alta', help='Nivel de complejidad')
    parser.add_argument('--api', type=str, choices=['openai', 'anthropic', 'local', 'gemini', 'deepseek', 'generic'], help='Proveedor de API')
    parser.add_argument('--base_path', type=str, default='.', help='Ruta base del proyecto')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    parser.add_argument('--model', type=str, help='Nombre del modelo a usar (útil para generic/local)')
    parser.add_argument('--base_url', type=str, help='URL base (útil para generic/local)')
    parser.add_argument('--examen_num', type=int, help='Número del examen')
    parser.add_argument('--no_generar_soluciones', action='store_true', help='NO generar soluciones')
    parser.add_argument('--subject', type=str, default='IF3602 - II semestre 2025', help='Asignatura')
    parser.add_argument('--keywords', type=str, nargs='+', help='Palabras clave')
    parser.add_argument('--mode', type=str, choices=['variation', 'creation'], default='variation', help='Modo de operación')
    parser.add_argument('--type', type=str, choices=['development', 'multiple_choice'], default='development', help='Tipo de ejercicio')
    parser.add_argument('--tags', type=str, nargs='+', help='Tags específicos para creación')
    parser.add_argument('--label', type=str, nargs='+', help='ID(s) específico(s) a variar')
    parser.add_argument('--use_rag', action='store_true', help='Usar RAG')
    parser.add_argument('--reindex', action='store_true', help='Forzar re-indexación RAG')
    parser.add_argument('--list', action='store_true', help='Listar ejercicios')
    parser.add_argument('--query', type=str, help='Consulta RAG')
    parser.add_argument('--workers', type=int, default=5, help='Número de hilos para generación paralela')
    parser.add_argument('--analyze', action='store_true', help='Analizar estructura y generar config.yaml')

    args = parser.parse_args()

    # Validar argumentos con ArgsValidator
    validator = ArgsValidator()
    is_valid, errors = validator.validate_args(args)

    if not is_valid:
        logger.error("Errores de validación en argumentos:")
        for error in errors:
            logger.error(f"  - {error}")
        parser.error("\n".join(errors))

    if args.analyze:
        print("Analizando estructura del proyecto...")
        ConfigManager(str(Path.cwd())).update_config_from_structure()
        return 0
    
    # Validaciones básicas de argumentos
    if not args.reindex and not args.list and not args.query:
        if not args.tema and not args.label:
            parser.error("--tema es requerido (excepto con --label, --reindex, --list o --query)")
        if not args.output:
            parser.error("--output es requerido")

    base_path = Path(args.base_path).resolve()
    if not base_path.exists():
        logger.error(f"Ruta base no existe: {base_path}")
        return 1

    # Inicializar Engine
    config_path = Path(args.config).resolve() if args.config else None
    engine = EvolutiaEngine(base_path, config_path)
    
    # Configurar API Provider Default si no se pasa
    if args.api is None:
        args.api = engine.full_config.get('api', {}).get('default_provider', 'openai')

    logger.info(f"Iniciando Evolutia (API: {args.api}, Mode: {args.mode})")

    try:
        # RAG Lifecycle
        if args.use_rag or args.query or args.reindex:
            success = engine.initialize_rag(force_reindex=args.reindex)
            if not success and (args.use_rag or args.query):
                 logger.error("Fallo crítico en inicialización RAG")
                 return 1
            if args.reindex and not args.tema: # Just reindex requested
                 logger.info("Reindexación completada.")
                 return 0

        # Query Mode
        if args.query:
             # Simple query logic inline or move to engine? Kept simple here using engine internals for now
             if engine.rag_manager:
                 logger.info(f"Consultando: '{args.query}'")
                 results = engine.rag_manager.get_retriever().hybrid_search(args.query)
                 print(f"\nResultados para '{args.query}':")
                 for i, res in enumerate(results, 1):
                     print(f"[{i}] {res.get('content')[:100]}... (Sim: {res.get('similarity'):.2f})")
                 return 0

        # 1. Extraction
        topics = args.tema if isinstance(args.tema, list) else ([args.tema] if args.tema else [])
        materials, all_exercises = engine.extract_materials_and_exercises(topics, args.label)
        
        if args.list:
            print(f"Ejercicios encontrados: {len(all_exercises)}")
            for ex in all_exercises:
                print(f"- {ex.get('label')}: {ex.get('source_file').name}")
            return 0
            
        if not all_exercises and args.mode != 'creation':
             logger.error("No hay ejercicios para procesar.")
             return 1

        # 2. RAG Indexing if needed (and using materials found)
        if args.use_rag and engine.rag_manager and not engine.rag_manager.is_indexed():
             engine.rag_manager.index_materials(materials, ExerciseAnalyzer())

        # 3. Analysis
        analyzed_exercises = engine.analyze_exercises(all_exercises)

        # 4. Generation (Parallel)
        variations = engine.generate_variations_parallel(analyzed_exercises, args, max_workers=args.workers)

        if not variations:
            logger.error("No se generaron variaciones válidas.")
            return 1
            
        # 5. Output
        output_dir = Path(args.output).resolve()
        # Determine exam num
        exam_num = args.examen_num if args.examen_num else 1
        # Simple heuristic if not provided
        if not args.examen_num and 'examen' in output_dir.name:
             import re
             idx = re.search(r'\d+', output_dir.name)
             if idx: exam_num = int(idx.group())

        success = engine.generate_exam_files(variations, args, output_dir, exam_num)
        
        if success:
            logger.info("Proceso finalizado con éxito.")
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario.")
        return 1
    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
