import logging
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class QdrantRetriever:
    def __init__(
        self,
        cloud_url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "my_documents",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Инициализация ретривера для Qdrant Cloud
        """
        # Получение данных из переменных окружения или параметров
        cloud_url = cloud_url or os.getenv("QDRANT_CLOUD_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not cloud_url or not api_key:
            raise ValueError("Необходимо указать cloud_url и api_key")
        
        # Инициализация клиента Qdrant Cloud
        self.client = QdrantClient(
            url=cloud_url,
            api_key=api_key,
            timeout=30
        )
        
        self.collection_name = collection_name
        
        # Загрузка модели для эмбеддингов
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Проверка подключения
        self._check_connection()
    
    def _check_connection(self):
        """Проверка подключения к Qdrant"""
        try:
            collections = self.client.get_collections()
            logger.info(f"Успешное подключение. Доступные коллекции: {[col.name for col in collections.collections]}")
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Поиск релевантных документов по запросу
        """
        try:
            # Создаем эмбеддинг для запроса
            logger.info(f"Создание эмбеддинга для запроса: '{query}'")
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Подготавливаем фильтр если есть
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            logger.info(f"Выполнение поиска в коллекции '{self.collection_name}'...")
            
            # ИСПРАВЛЕНО для qdrant-client >= 1.13: используем query_points с параметром query
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,  # НЕ query_vector, а query!
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            
            # Форматируем результаты - результат в .points
            results = []
            for hit in search_result.points:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'content': hit.payload.get('content', ''),
                    'filename': hit.payload.get('filename', ''),
                    'document_id': hit.payload.get('document_id', ''),
                    'chunk_index': hit.payload.get('chunk_index', 0),
                    'total_chunks': hit.payload.get('total_chunks', 0),
                    'metadata': hit.payload.get('metadata', {})
                }
                results.append(result)
            
            # Сортируем по score (по убыванию)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Найдено {len(results)} релевантных чанков")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _build_filter(self, conditions: Dict) -> models.Filter:
        """
        Создание фильтра для поиска
        """
        filter_conditions = []
        
        for key, value in conditions.items():
            if isinstance(value, list):
                # Для фильтрации по массиву значений
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                # Для точного совпадения
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=filter_conditions)
    
    def get_collection_info(self) -> Dict:
        """
        Получение информации о коллекции
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            return {}
    
    def count_points(self) -> int:
        """
        Получение количества точек в коллекции
        """
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            return count_result.count
        except Exception as e:
            logger.error(f"Ошибка при подсчете точек: {e}")
            return 0
    
    def scroll_all_points(self, limit: int = 100) -> List[Dict]:
        """
        Просмотр всех точек в коллекции
        """
        try:
            all_points = []
            next_page_offset = None
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points = scroll_result[0]
                next_page_offset = scroll_result[1]
                
                for point in points:
                    all_points.append({
                        'id': point.id,
                        'payload': point.payload,
                        'score': None  # нет скора при скролле
                    })
                
                if next_page_offset is None:
                    break
            
            return all_points
        except Exception as e:
            logger.error(f"Ошибка при скролле точек: {e}")
            return []


# Альтернативная версия - простой поиск без класса
def simple_search():
    """Простой поиск без использования класса"""
    
    # Получаем параметры из .env
    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not cloud_url or not api_key:
        print("Ошибка: Не заданы QDRANT_CLOUD_URL и QDRANT_API_KEY в .env файле")
        return
    
    # Подключаемся к Qdrant
    client = QdrantClient(
        url=cloud_url,
        api_key=api_key,
        timeout=30
    )
    
    # Загружаем модель для эмбеддингов
    print("Загрузка модели для эмбеддингов...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Запрашиваем поисковый запрос
    query = input("Введите ваш запрос: ").strip()
    if not query:
        query = "субсидирование"
    
    print(f"\nПоиск: '{query}'")
    
    # Создаем эмбеддинг
    query_embedding = model.encode(query).tolist()
    
    try:
        # ИСПРАВЛЕНО для qdrant-client >= 1.13
        # Параметр называется query, НЕ query_vector!
        search_result = client.query_points(
            collection_name="my_documents",
            query=query_embedding,  # <-- ВАЖНО: query, а не query_vector
            limit=5,
            with_payload=True,
        )
        
        print(f"\nНайдено результатов: {len(search_result.points)}\n")
        
        for i, hit in enumerate(search_result.points, 1):
            print(f"Результат {i}:")
            print(f"  Файл: {hit.payload.get('filename', 'Неизвестно')}")
            print(f"  Схожесть: {hit.score:.3f}")
            print(f"  Контент: {hit.payload.get('content', '')[:200]}...")
            print()
            
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        import traceback
        traceback.print_exc()


# Интерфейс командной строки
def interactive_cli():
    """Интерактивный режим командной строки"""
    
    print("🤖 RAG Поиск по документам")
    print("Команды: 'exit' - выход, 'info' - информация, 'all' - все документы")
    print("-" * 50)
    
    # Инициализация ретривера
    try:
        retriever = QdrantRetriever(collection_name="my_documents")
        print("✅ Подключение к Qdrant Cloud успешно!")
        
        # Показываем информацию о коллекции
        info = retriever.get_collection_info()
        print(f"📊 Коллекция: {info.get('name')}")
        print(f"📊 Точек в коллекции: {info.get('points_count', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print("Пробуем простой режим...")
        simple_search()
        return
    
    while True:
        command = input("\nВведите запрос или команду: ").strip()
        
        if command.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break
        
        elif command.lower() == 'info':
            info = retriever.get_collection_info()
            print(f"Коллекция: {info.get('name')}")
            print(f"Точек: {info.get('points_count')}")
        
        elif command.lower() == 'all':
            limit = input("Сколько показать? (по умолчанию 5): ").strip()
            limit = int(limit) if limit.isdigit() else 5
            
            points = retriever.scroll_all_points(limit=limit)
            print(f"\nПоказано {len(points)} документов:")
            
            for i, point in enumerate(points, 1):
                payload = point['payload']
                print(f"\n{i}. {payload.get('filename', 'Без имени')}")
                print(f"   Чанк: {payload.get('chunk_index')}/{payload.get('total_chunks')}")
                print(f"   {payload.get('content', '')[:100]}...")
        
        else:
            # Обычный поиск
            limit_input = input("Количество результатов (по умолчанию 3): ").strip()
            limit = int(limit_input) if limit_input.isdigit() else 3
            
            print(f"\nПоиск: '{command}'...")
            results = retriever.search(command, limit=limit)
            
            if results:
                print(f"Найдено {len(results)} результатов:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. 📄 {result['filename']} (схожесть: {result['score']:.3f})")
                    print(f"   {result['content'][:150]}...")
                    print()
            else:
                print("Результатов не найдено")


if __name__ == "__main__":
    # Простой режим
    simple_search()
    
    # Или интерактивный режим
    # interactive_cli()