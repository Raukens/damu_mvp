import os
import sys
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI
from dotenv import load_dotenv


class RAGSystem:
    def __init__(self):
        """
        RAG система, которая всегда берет ключи из переменных окружения
        """
        # Получаем ключи ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
        load_dotenv()
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.openai_api_key = os.getenv("GPT_KEY")
        self.collection_name = "my_documents"
        
        # Проверяем наличие всех обязательных ключей
        self._validate_environment_variables()
        
        # Инициализация клиентов
        self._init_clients()
        
    def _validate_environment_variables(self):
        """Проверка наличия всех необходимых переменных окружения"""
        missing_vars = []
        
        if not self.qdrant_url:
            missing_vars.append("QDRANT_URL")
        if not self.qdrant_api_key:
            missing_vars.append("QDRANT_API_KEY")
        if not self.openai_api_key:
            missing_vars.append("GPT_KEY")
        
        if missing_vars:
            error_msg = f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}\n"
            error_msg += "Установите их командой:\n"
            error_msg += "  Linux/Mac: export ИМЯ_ПЕРЕМЕННОЙ='значение'\n"
            error_msg += "  Windows PowerShell: $env:ИМЯ_ПЕРЕМЕННОЙ='значение'\n"
            error_msg += "  Windows CMD: set ИМЯ_ПЕРЕМЕННОЙ=значение\n"
            raise EnvironmentError(error_msg)
    
    def _init_clients(self):
        """Инициализация клиентов Qdrant и OpenAI"""
        try:
            print(f"🔗 Подключение к Qdrant...")
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=30
            )
            
            print(f"🤖 Подключение к OpenAI...")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            
            # Проверяем соединение с Qdrant
            collections = self.qdrant_client.get_collections()
            print(f"✓ Подключено к Qdrant. Коллекции: {[c.name for c in collections.collections]}")
            
            # Проверяем коллекцию
            if self.collection_name not in [c.name for c in collections.collections]:
                print(f"⚠️  Коллекция '{self.collection_name}' не найдена")
                print("Доступные коллекции:", [c.name for c in collections.collections])
            
            print(f"✓ RAG система инициализирована")
            print(f"✓ Используется коллекция: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            raise
    
    def _get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Получение эмбеддинга для текста
        """
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Ошибка при получении эмбеддинга: {e}")
            raise
    
    def search_documents(self, query: str, limit: int = 5) -> List[dict]:
        """
        Поиск документов, релевантных запросу
        """
        try:
            # Векторизуем запрос
            query_embedding = self._get_embedding(query)
            
            # Ищем в Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Форматируем результаты
            documents = []
            for result in search_results:
                doc = {
                    "id": str(result.id),
                    "score": float(result.score),
                    "content": result.payload.get("content", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "content"}
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Ошибка при поиске документов: {e}")
            return []
    
    def generate_answer(self, question: str, documents: List[dict]) -> str:
        """
        Генерация ответа на основе найденных документов
        """
        if not documents:
            return "Извините, не нашел подходящей информации в документах."
        
        # Формируем контекст из документов
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Берем только начало документа для экономии токенов
            content_preview = doc['content'][:800] + "..." if len(doc['content']) > 800 else doc['content']
            context_parts.append(f"[Документ {i}, релевантность: {doc['score']:.3f}]:\n{content_preview}")
        
        context = "\n\n".join(context_parts)
        
        # Создаем промпт
        prompt = f"""Ты - AI-ассистент, который отвечает на вопросы на основе предоставленных документов.

Контекст из документов:
{context}

Вопрос: {question}

Инструкции:
1. Ответь строго на основе предоставленных документов
2. Если информации в документах недостаточно, скажи об этом
3. Будь точным и конкретным
4. Не придумывай информацию, которой нет в документах

Ответ:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Можно заменить на gpt-4 если нужно
                messages=[
                    {"role": "system", "content": "Ты полезный ассистент, который отвечает на вопросы на основе документов."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def ask(self, question: str, show_sources: bool = True) -> dict:
        """
        Основной метод для получения ответа на вопрос
        
        Возвращает:
        {
            "answer": "текст ответа",
            "sources": [список источников],
            "documents_found": количество найденных документов
        }
        """
        print(f"\n{'='*60}")
        print(f"🤔 Вопрос: {question}")
        
        # 1. Поиск документов
        print("🔍 Ищу релевантные документы...")
        documents = self.search_documents(question, limit=3)
        
        if not documents:
            return {
                "answer": "В моей базе знаний нет информации по этому вопросу.",
                "sources": [],
                "documents_found": 0
            }
        
        print(f"✅ Найдено {len(documents)} документов")
        
        # 2. Генерация ответа
        print("🤖 Генерирую ответ...")
        answer = self.generate_answer(question, documents)
        
        # 3. Подготовка информации об источниках
        sources = []
        if show_sources:
            for doc in documents:
                source_info = {
                    "id": doc["id"],
                    "relevance": round(doc["score"], 3),
                    "preview": doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"],
                    "metadata": doc["metadata"]
                }
                sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "documents_found": len(documents)
        }
    
    def test_connection(self) -> bool:
        """
        Тестирование подключения ко всем сервисам
        """
        print("🧪 Тестирование подключений...")
        
        try:
            # Тест OpenAI
            models = self.openai_client.models.list()
            print(f"✓ OpenAI: OK (доступно моделей: {len(list(models))})")
            
            # Тест Qdrant
            collections = self.qdrant_client.get_collections()
            print(f"✓ Qdrant: OK (коллекций: {len(collections.collections)})")
            
            # Проверка коллекции
            if self.collection_name in [c.name for c in collections.collections]:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                print(f"✓ Коллекция '{self.collection_name}': OK (точек: {collection_info.points_count})")
            else:
                print(f"⚠️  Коллекция '{self.collection_name}' не найдена")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")
            return False


def print_env_info():
    """
    Вывод информации о переменных окружения
    """
    print("📋 Информация о переменных окружения:")
    print("-" * 50)
    
    env_vars = {
        "QDRANT_CLOUD_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "GPT_KEY": os.getenv("GPT_KEY"),
        "COLLECTION_NAME": 'my_documents',
    }
    
    for name, value in env_vars.items():
        if value:
            # Маскируем длинные ключи для безопасности
            if "KEY" in name:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {name}: {masked}")
            else:
                print(f"  {name}: {value}")
        else:
            print(f"  {name}: ❌ НЕ ЗАДАНА")
    
    print("-" * 50)


def interactive_mode():
    """
    Интерактивный режим работы с системой
    """
    print("\n" + "="*60)
    print("🚀 RAG Система")
    print("="*60)
    
    # Показываем информацию о переменных окружения
    print_env_info()
    
    try:
        # Инициализация системы
        rag = RAGSystem()
        
        # Тестируем подключение
        if not rag.test_connection():
            print("❌ Проблемы с подключением. Проверьте настройки.")
            return
        
        print("\n✅ Система готова к работе!")
        print("Команды:")
        print("  - Просто задавайте вопросы")
        print("  - 'тест' - проверить подключение")
        print("  - 'источники' - вкл/выкл показ источников")
        print("  - 'выход' или 'quit' - завершить работу")
        print("-" * 60)
        
        show_sources = True
        
        while True:
            try:
                # Получаем вопрос
                user_input = input("\n🧑 Ваш вопрос: ").strip()
                
                # Обработка команд
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("👋 До свидания!")
                    break
                
                if user_input.lower() == 'тест':
                    rag.test_connection()
                    continue
                
                if user_input.lower() == 'источники':
                    show_sources = not show_sources
                    status = "ВКЛЮЧЕН" if show_sources else "ВЫКЛЮЧЕН"
                    print(f"📚 Показ источников: {status}")
                    continue
                
                if not user_input:
                    continue
                
                # Получаем ответ
                result = rag.ask(user_input, show_sources=show_sources)
                
                # Выводим результат
                print(f"\n{'='*60}")
                print("🤖 Ответ:")
                print(result["answer"])
                
                # Показываем источники если нужно
                if show_sources and result["sources"]:
                    print(f"\n📚 Источники ({result['documents_found']} шт):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"\n{i}. ID: {source['id']}")
                        print(f"   Релевантность: {source['relevance']}")
                        if source['metadata']:
                            print(f"   Метаданные: {source['metadata']}")
                        print(f"   Фрагмент: {source['preview']}")
                
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Завершение работы...")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
    
    except EnvironmentError as e:
        print(f"\n{e}")
        print("\n💡 Совет: Установите переменные окружения:")
        print("""
  # Linux/Mac:
  export QDRANT_CLOUD_URL='ваш_url_из_qdrant_cloud'
  export QDRANT_API_KEY='ваш_api_key_из_qdrant'
  export GPT_KEY='sk-ваш_ключ_openai'
  export COLLECTION_NAME='documents'  # опционально

  # Windows PowerShell:
  $env:QDRANT_CLOUD_URL='ваш_url_из_qdrant_cloud'
  $env:QDRANT_API_KEY='ваш_api_key_из_qdrant'
  $env:GPT_KEY='sk-ваш_ключ_openai'
  $env:COLLECTION_NAME='documents'  # опционально
        """)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


def quick_test_mode():
    """
    Режим быстрого тестирования
    """
    print("🧪 Быстрый тест системы...")
    
    try:
        rag = RAGSystem()
        
        # Тестовые вопросы
        test_questions = [
            "Какая основная тема документов?",
            "Что содержится в документах?",
            "Расскажи кратко о содержании",
        ]
        
        for question in test_questions:
            print(f"\n📝 Тестовый вопрос: {question}")
            result = rag.ask(question, show_sources=False)
            print(f"📤 Ответ: {result['answer'][:200]}...")
            print(f"📄 Найдено документов: {result['documents_found']}")
            print("-" * 60)
    
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")


if __name__ == "__main__":
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            quick_test_mode()
        elif command == "env":
            print_env_info()
        elif command == "help":
            print("""
Использование:
  python rag_system.py           # Интерактивный режим (по умолчанию)
  python rag_system.py test     # Быстрый тест
  python rag_system.py env      # Показать переменные окружения
  python rag_system.py help     # Эта справка
  
Переменные окружения (ОБЯЗАТЕЛЬНЫ):
  QDRANT_CLOUD_URL    - URL вашего Qdrant Cloud кластера
  QDRANT_API_KEY      - API ключ Qdrant
  GPT_KEY             - API ключ OpenAI
  
Переменные окружения (опционально):
  COLLECTION_NAME     - Название коллекции (по умолчанию: 'documents')
            """)
        else:
            print(f"❌ Неизвестная команда: {command}")
            print("Используйте: python rag_system.py [test|env|help]")
    else:
        # Запуск в интерактивном режиме по умолчанию
        interactive_mode()