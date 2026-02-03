"""
Caché de análisis de ejercicios para EvolutIA.
Reduce tiempo de ejecución almacenando análisis de ejercicios.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ExerciseAnalysisCache:
    """
    Sistema de caché para análisis de ejercicios.

    Características:
    - Persistencia en disco por defecto
    - Basado en hash del contenido del ejercicio
    - Valida integridad del caché
    - Logging de cache hits y misses
    """

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        Inicializa el caché de análisis de ejercicios.

        Args:
            cache_dir: Directorio para caché (defecto: ./storage/cache/exercises)
            enabled: Si False, el caché está deshabilitado (pasa a través)
        """
        self.enabled = enabled

        if cache_dir is None:
            cache_dir = Path('./storage/cache/exercises')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hits = 0
        self.misses = 0

        if self.enabled:
            logger.info(f"[ExerciseAnalysisCache] Inicializado: {self.cache_dir}")
        else:
            logger.info("[ExerciseAnalysisCache] Deshabilitado")

    def _get_cache_file(self, content_hash: str) -> Path:
        """
        Obtiene la ruta del archivo de caché para un hash.

        Args:
            content_hash: Hash del contenido del ejercicio

        Returns:
            Ruta del archivo de caché
        """
        return self.cache_dir / f"{content_hash}.json"

    def _hash_content(self, content: str) -> str:
        """
        Genera un hash SHA256 del contenido.

        Args:
            content: Contenido del ejercicio

        Returns:
            Hash SHA256 hexadecimal
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, exercise: Dict) -> Optional[Dict]:
        """
        Obtiene el análisis cacheado de un ejercicio.

        Args:
            exercise: Diccionario del ejercicio con campo 'content'

        Returns:
            Análisis cacheado si existe, None en caso contrario
        """
        if not self.enabled:
            return None

        content = exercise.get('content', '')
        if not content:
            self.misses += 1
            return None

        content_hash = self._hash_content(content)
        cache_file = self._get_cache_file(content_hash)

        if not cache_file.exists():
            self.misses += 1
            logger.debug("[ExerciseAnalysisCache] Cache miss")
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)

            self.hits += 1
            logger.info(
                f"[ExerciseAnalysisCache] Cache HIT "
                f"(hit_rate={self.hit_rate:.1%})"
            )
            return analysis
        except Exception as e:
            logger.warning(f"[ExerciseAnalysisCache] Error leyendo caché: {e}")
            self.misses += 1
            return None

    def put(self, exercise: Dict, analysis: Dict) -> bool:
        """
        Almacena el análisis de un ejercicio en caché.

        Args:
            exercise: Diccionario del ejercicio con campo 'content'
            analysis: Análisis del ejercicio

        Returns:
            True si se almacenó exitosamente, False si hubo error
        """
        if not self.enabled:
            return False

        content = exercise.get('content', '')
        if not content:
            logger.debug("[ExerciseAnalysisCache] Rechazando ejercicio sin contenido")
            return False

        # Validar que el análisis tiene los campos mínimos
        required_fields = ['total_complexity']
        if not all(field in analysis for field in required_fields):
            logger.warning("[ExerciseAnalysisCache] Análisis incompleto, no cachéando")
            return False

        content_hash = self._hash_content(content)
        cache_file = self._get_cache_file(content_hash)

        try:
            # Almacenar análisis con metadata
            cache_data = {
                'analysis': analysis,
                'metadata': {
                    'cached_at': None,  # Se llenará después
                    'content_length': len(content),
                    'exercise_label': exercise.get('label', 'unknown')
                }
            }

            import time
            cache_data['metadata']['cached_at'] = time.time()

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"[ExerciseAnalysisCache] Análisis cachéado: {cache_file.name}")
            return True
        except Exception as e:
            logger.warning(f"[ExerciseAnalysisCache] Error guardando caché: {e}")
            return False

    def clear(self):
        """Limpia todo el caché de ejercicios."""
        if not self.enabled:
            return

        initial_count = len(list(self.cache_dir.glob('*.json')))

        for cache_file in self.cache_dir.glob('*.json'):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"[ExerciseAnalysisCache] Error eliminando {cache_file}: {e}")

        self.hits = 0
        self.misses = 0

        logger.info(f"[ExerciseAnalysisCache] Caché limpiado (eliminados {initial_count} archivos)")

    def get_stats(self) -> Dict[str, any]:
        """
        Obtiene estadísticas del caché.

        Returns:
            Diccionario con estadísticas
        """
        cache_files = list(self.cache_dir.glob('*.json'))
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            'entries': len(cache_files),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'enabled': self.enabled,
            'cache_dir': str(self.cache_dir)
        }

    @property
    def hit_rate(self) -> float:
        """
        Tasa de aciertos del caché.

        Returns:
            Proporción de aciertos (0.0 a 1.0)
        """
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0

    def __len__(self) -> int:
        """Retorna el número de entradas en caché."""
        if not self.enabled:
            return 0
        return len(list(self.cache_dir.glob('*.json')))

    def __repr__(self) -> str:
        """Representación del caché."""
        status = "enabled" if self.enabled else "disabled"
        return (
            f"ExerciseAnalysisCache(status={status}, "
            f"entries={len(self)}, hits={self.hits}, "
            f"misses={self.misses}, hit_rate={self.hit_rate:.1%})"
        )
