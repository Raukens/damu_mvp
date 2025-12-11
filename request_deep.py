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
        
        # Проверяем версию API
        self._check_api_version()
    
    def _check_connection(self):
        """Проверка подключения к Qdrant"""
        try:
            collections = self.client.get_collections()
            logger.info(f"Успешное подключение. Доступные коллекции: {[col.name for col in collections.collections]}")
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            raise
    
    def _check_api_version(self):
        """Проверяем доступные методы API"""
        self.use_search_points = hasattr(self.client, 'search_points')
        self.use_query_points = hasattr(self.client, 'query_points')
        self.use_search = hasattr(self.client, 'search')
        
        logger.info(f"API методы доступны: search_points={self.use_search_points}, "
                   f"query_points={self.use_query_points}, search={self.use_search}")
    
    def search(
        self,
        query: str,
        limit: int = 10,
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
            search_filter = None
            if filter_conditions:
                search_filter = self._build_filter(filter_conditions)
            
            logger.info(f"Выполнение поиска в коллекции '{self.collection_name}'...")
            
            # Пробуем разные методы в зависимости от доступности
            search_result = None
            
            # Метод 1: используем search (самый новый)
            if self.use_search:
                logger.info("Использую метод: client.search()")
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=search_filter,
                    score_threshold=score_threshold
                )
            
            # Метод 2: используем search_points (старый API)
            elif self.use_search_points:
                logger.info("Использую метод: client.search_points()")
                from qdrant_client.http.models import SearchRequest
                
                search_request = SearchRequest(
                    vector=query_embedding,
                    limit=limit,
                    filter=search_filter,
                    score_threshold=score_threshold
                )
                
                search_response = self.client.search_points(
                    collection_name=self.collection_name,
                    search_request=search_request
                )
                search_result = search_response.result
            
            # Метод 3: используем query_points (другой API)
            elif self.use_query_points:
                logger.info("Использую метод: client.query_points()")
                search_params = {
                    "collection_name": self.collection_name,
                    "query": query_embedding,
                    "limit": limit,
                    "with_payload": True,
                    "with_vectors": False
                }
                
                if search_filter:
                    search_params["filter"] = search_filter
                
                if score_threshold is not None:
                    search_params["score_threshold"] = score_threshold
                
                response = self.client.query_points(**search_params)
                search_result = response.points
            
            else:
                raise Exception("Нет доступных методов поиска. Проверьте версию qdrant-client")
            
            # Форматируем результаты
            results = self._format_search_results(search_result)
            
            logger.info(f"Найдено {len(results)} релевантных чанков")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []
    
    def _format_search_results(self, search_result) -> List[Dict]:
        """Форматирование результатов поиска в единый формат"""
        results = []
        
        # Если результат в виде списка (новый API)
        if isinstance(search_result, list):
            for hit in search_result:
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
        
        # Если результат в виде объекта с точками (старый API)
        elif hasattr(search_result, 'points'):
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
        
        # Если результат в виде объекта с result (очень старый API)
        elif hasattr(search_result, 'result'):
            for hit in search_result.result:
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
        return results
    
    def _build_filter(self, conditions: Dict) -> Optional[models.Filter]:
        """
        Создание фильтра для поиска
        """
        if not conditions:
            return None
            
        filter_conditions = []
        
        for key, value in conditions.items():
            if isinstance(value, list):
                # Для фильтрации по массиву значений
                filter_conditions.append(
                    models.FieldCondition(
                        key=f"payload.{key}",
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                # Для точного совпадения
                filter_conditions.append(
                    models.FieldCondition(
                        key=f"payload.{key}",
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=filter_conditions) if filter_conditions else None
    
    def get_collection_info(self) -> Dict:
        """
        Получение информации о коллекции
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': collection_info.name,
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
    
    # Проверяем доступные методы
    use_search_points = hasattr(client, 'search_points')
    use_query_points = hasattr(client, 'query_points')
    use_search = hasattr(client, 'search')
    
    print(f"Доступные методы: search_points={use_search_points}, query_points={use_query_points}, search={use_search}")
    
    # Загружаем модель для эмбеддингов
    print("\nЗагрузка модели для эмбеддингов...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Запрашиваем поисковый запрос
    query = input("Введите ваш запрос: ").strip()
    if not query:
        query = "субсидирование"
    
    print(f"\nПоиск: '{query}'")
    
    # Создаем эмбеддинг
    query_embedding = model.encode(query).tolist()
    
    try:
        # Пробуем разные методы поиска
        search_result = None
        
        if use_search_points:
            print("Использую метод: search_points")
            from qdrant_client.http.models import SearchRequest
            
            search_request = SearchRequest(
                vector=query_embedding,
                limit=5,
                with_payload=True,
                with_vector=False
            )
            
            response = client.search_points(
                collection_name="my_documents",
                search_request=search_request
            )
            search_result = response.result
        
        elif use_search:
            print("Использую метод: search")
            search_result = client.search(
                collection_name="my_documents",
                query_vector=query_embedding,
                limit=5
            )
        
        elif use_query_points:
            print("Использую метод: query_points")
            response = client.query_points(
                collection_name="my_documents",
                query=query_embedding,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            search_result = response.points
        
        else:
            print("❌ Нет доступных методов поиска")
            return
        
        # Обрабатываем результаты
        if hasattr(search_result, 'result'):
            hits = search_result.result
        elif hasattr(search_result, 'points'):
            hits = search_result.points
        elif isinstance(search_result, list):
            hits = search_result
        else:
            hits = []
        
        print(f"\n✅ Найдено результатов: {len(hits)}\n")
        
        for i, hit in enumerate(hits, 1):
            print(f"Результат {i}:")
            print(f"  ID: {hit.id}")
            print(f"  Файл: {hit.payload.get('filename', 'Неизвестно')}")
            print(f"  Схожесть: {hit.score:.3f}")
            content = hit.payload.get('content', '')
            print(f"  Контент: {content[:200]}..." if len(content) > 200 else f"  Контент: {content}")
            
            # Дополнительная информация если есть
            if 'chunk_index' in hit.payload:
                print(f"  Чанк: {hit.payload.get('chunk_index')}/{hit.payload.get('total_chunks', 1)}")
            print()
            
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        
        # Пробуем получить информацию о коллекции
        try:
            collections = client.get_collections()
            print(f"\nДоступные коллекции: {[col.name for col in collections.collections]}")
            
            # Пробуем другую коллекцию если есть
            available_collections = [col.name for col in collections.collections]
            if "my_documents" not in available_collections and available_collections:
                print(f"\nКоллекция 'my_documents' не найдена. Попробуйте использовать: {available_collections[0]}")
        except Exception as e2:
            print(f"Не удалось получить список коллекций: {e2}")

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
            print(f"📊 Коллекция: {info.get('name', 'N/A')}")
            print(f"📊 Точек: {info.get('points_count', 'N/A')}")
            print(f"📊 Статус: {info.get('status', 'N/A')}")
        
        elif command.lower() == 'all':
            limit = input("Сколько показать? (по умолчанию 5): ").strip()
            limit = int(limit) if limit.isdigit() else 5
            
            points = retriever.scroll_all_points(limit=limit)
            print(f"\nПоказано {len(points)} документов:")
            
            for i, point in enumerate(points, 1):
                payload = point['payload']
                print(f"\n{i}. 📄 {payload.get('filename', 'Без имени')}")
                if 'chunk_index' in payload:
                    print(f"   Чанк: {payload.get('chunk_index')}/{payload.get('total_chunks', 1)}")
                content = payload.get('content', '')
                print(f"   {content[:100]}..." if len(content) > 100 else f"   {content}")
        
        else:
            # Обычный поиск
            limit_input = input("Количество результатов (по умолчанию 3): ").strip()
            limit = int(limit_input) if limit_input.isdigit() else 3
            
            print(f"\n🔍 Поиск: '{command}'...")
            results = retriever.search(command, limit=limit)
            
            if results:
                print(f"✅ Найдено {len(results)} результатов:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. 📄 {result['filename']} (схожесть: {result['score']:.3f})")
                    if result.get('chunk_index'):
                        print(f"   📑 Чанк: {result['chunk_index']}/{result['total_chunks']}")
                    content = result['content']
                    print(f"   {content[:150]}..." if len(content) > 150 else f"   {content}")
                    print()
            else:
                print("❌ Результатов не найдено")
                print("Проверьте:")
                print("1. Существует ли коллекция 'my_documents'")
                print("2. Загружены ли документы в коллекцию")

if __name__ == "__main__":
    # Сначала проверяем версию qdrant-client
    try:
        import qdrant_client
        print(f"Версия qdrant-client: {qdrant_client.__version__}")
    except:
        print("Не удалось определить версию qdrant-client")
    
    # Простой режим
    simple_search()
    
    # Или интерактивный режим
    # interactive_cli()