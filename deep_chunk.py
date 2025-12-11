import os
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantCloudIndexer:
    def __init__(
        self,
        cloud_url: str,
        api_key: str,
        collection_name: str = "documents",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Инициализация индексатора для Qdrant Cloud
        
        Args:
            cloud_url: URL вашего Qdrant Cloud инстанса
            api_key: API ключ для аутентификации
            collection_name: Название коллекции
            embedding_model_name: Название модели для эмбеддингов
        """
        # Инициализация клиента Qdrant Cloud
        self.client = QdrantClient(
            url=cloud_url,
            api_key=api_key,
            timeout=60  # Увеличиваем таймаут для больших файлов
        )
        
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Загрузка модели для эмбеддингов
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Получаем размерность модели
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Размерность вектора: {self.vector_size}")
    
    def check_connection(self) -> bool:
        """
        Проверка подключения к Qdrant Cloud
        
        Returns:
            bool: True если подключение успешно
        """
        try:
            collections = self.client.get_collections()
            logger.info(f"Успешное подключение к Qdrant Cloud. Коллекции: {collections}")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant Cloud: {e}")
            return False
    
    def create_collection(self, recreate_if_exists: bool = False):
        """
        Создание коллекции в Qdrant Cloud
        
        Args:
            recreate_if_exists: Удалить и пересоздать если коллекция существует
        """
        try:
            # Проверяем существование коллекции
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if collection_exists:
                if recreate_if_exists:
                    logger.info(f"Удаление существующей коллекции: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Коллекция '{self.collection_name}' уже существует")
                    return
            
            # Создаем новую коллекцию
            logger.info(f"Создание коллекции '{self.collection_name}' с размерностью {self.vector_size}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2  # Оптимизация для облака
                )
            )
            
            logger.info(f"Коллекция '{self.collection_name}' успешно создана")
            
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            raise
    
    def read_documents(self, docs_folder: str, file_extensions: Optional[List[str]] = None) -> List[dict]:
        """
        Чтение документов из папки
        
        Args:
            docs_folder: Путь к папке с документами
            file_extensions: Список поддерживаемых расширений
            
        Returns:
            Список документов с метаданными
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf', '.docx', '.doc']
        
        documents = []
        
        if not os.path.exists(docs_folder):
            raise ValueError(f"Папка '{docs_folder}' не существует")
        
        logger.info(f"Чтение документов из папки: {docs_folder}")
        logger.info(f"Поддерживаемые расширения: {file_extensions}")
        
        for filename in tqdm(os.listdir(docs_folder), desc="Чтение файлов"):
            file_path = os.path.join(docs_folder, filename)
            
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                
                if ext.lower() in file_extensions:
                    try:
                        # Чтение содержимого файла
                        content = self._read_file_content(file_path, ext.lower())
                        
                        if content:
                            document = {
                                'id': str(uuid.uuid4()),
                                'filename': filename,
                                'content': content,
                                'file_path': file_path,
                                'file_size': os.path.getsize(file_path),
                                'file_extension': ext.lower()
                            }
                            
                            documents.append(document)
                            
                    except Exception as e:
                        logger.error(f"Ошибка при чтении файла {filename}: {e}")
        
        logger.info(f"Загружено {len(documents)} документов")
        return documents
    
    def _read_file_content(self, file_path: str, extension: str) -> str:
        """
        Чтение содержимого файла в зависимости от расширения
        
        Args:
            file_path: Путь к файлу
            extension: Расширение файла
            
        Returns:
            Текст содержимого файла
        """
        try:
            if extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif extension == '.pdf':
                return self._read_pdf(file_path)
            
            elif extension in ['.docx', '.doc']:
                return self._read_docx(file_path)
            
            else:
                logger.warning(f"Неизвестное расширение: {extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return ""
    
    def _read_pdf(self, file_path: str) -> str:
        """
        Чтение PDF файла
        
        Args:
            file_path: Путь к PDF файлу
            
        Returns:
            Текст из PDF
        """
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            return text
        except ImportError:
            logger.error("Для чтения PDF установите PyPDF2: pip install PyPDF2")
            return ""
    
    def _read_docx(self, file_path: str) -> str:
        """
        Чтение DOCX файла
        
        Args:
            file_path: Путь к DOCX файлу
            
        Returns:
            Текст из DOCX
        """
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.error("Для чтения DOCX установите python-docx: pip install python-docx")
            return ""
    
    def chunk_documents(
        self,
        documents: List[dict],
        chunk_size: int = 1000,
        overlap: int = 200,
        chunk_by: str = "characters"  # characters, sentences, paragraphs
    ) -> List[dict]:
        """
        Разбивка документов на чанки
        
        Args:
            documents: Список документов
            chunk_size: Размер чанка
            overlap: Перекрытие между чанками
            chunk_by: Метод разбивки (characters, sentences, paragraphs)
            
        Returns:
            Список чанков
        """
        chunks = []
        
        logger.info(f"Разбивка документов на чанки (метод: {chunk_by})")
        
        for doc in tqdm(documents, desc="Разбивка документов"):
            content = doc['content']
            
            if chunk_by == "characters":
                doc_chunks = self._chunk_by_characters(content, chunk_size, overlap)
            elif chunk_by == "sentences":
                doc_chunks = self._chunk_by_sentences(content, chunk_size, overlap)
            elif chunk_by == "paragraphs":
                doc_chunks = self._chunk_by_paragraphs(content, chunk_size)
            else:
                doc_chunks = self._chunk_by_characters(content, chunk_size, overlap)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk = {
                    'id': str(uuid.uuid4()),
                    'document_id': doc['id'],
                    'filename': doc['filename'],
                    'content': chunk_text,
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks),
                    'file_extension': doc.get('file_extension', ''),
                    'metadata': {
                        'filename': doc['filename'],
                        'document_id': doc['id']
                    }
                }
                chunks.append(chunk)
        
        logger.info(f"Создано {len(chunks)} чанков из {len(documents)} документов")
        return chunks
    
    def _chunk_by_characters(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Разбивка по символам"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Находим границу предложения для более чистого разделения
            if end < len(text):
                # Ищем ближайшую точку, пробел или перенос строки
                while end > start and text[end] not in ['.', ' ', '\n', '!', '?']:
                    end -= 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end - overlap > start else end
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, max_sentences: int, overlap_sentences: int) -> List[str]:
        """Разбивка по предложениям"""
        import re
        
        # Простое разделение на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        start = 0
        
        while start < len(sentences):
            end = min(start + max_sentences, len(sentences))
            chunk = ' '.join(sentences[start:end])
            chunks.append(chunk)
            
            start = end - overlap_sentences if end - overlap_sentences > start else end
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, max_paragraphs: int) -> List[str]:
        """Разбивка по абзацам"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        
        for i in range(0, len(paragraphs), max_paragraphs):
            chunk = '\n\n'.join(paragraphs[i:i + max_paragraphs])
            chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, chunks: List[dict], batch_size: int = 32) -> List[dict]:
        """
        Создание эмбеддингов для чанков
        
        Args:
            chunks: Список чанков
            batch_size: Размер батча для обработки
            
        Returns:
            Список чанков с эмбеддингами
        """
        logger.info("Создание эмбеддингов...")
        
        # Извлекаем тексты из чанков
        texts = [chunk['content'] for chunk in chunks]
        
        # Создаем эмбеддинги батчами
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Создание эмбеддингов"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True  # Нормализация для косинусного расстояния
            )
            embeddings.extend(batch_embeddings)
        
        # Добавляем эмбеддинги к чанкам
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks
    
    def index_to_qdrant(self, chunks_with_embeddings: List[dict], batch_size: int = 100):
        """
        Индексация чанков в Qdrant Cloud
        
        Args:
            chunks_with_embeddings: Список чанков с эмбеддингами
            batch_size: Размер батча для загрузки
        """
        logger.info(f"Загрузка {len(chunks_with_embeddings)} чанков в Qdrant Cloud...")
        
        # Разбиваем на батчи для загрузки
        for i in tqdm(range(0, len(chunks_with_embeddings), batch_size), desc="Загрузка в Qdrant"):
            batch_chunks = chunks_with_embeddings[i:i + batch_size]
            
            points = []
            for chunk in batch_chunks:
                point = models.PointStruct(
                    id=chunk['id'],
                    vector=chunk['embedding'],
                    payload={
                        'content': chunk['content'],
                        'filename': chunk['filename'],
                        'document_id': chunk['document_id'],
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks'],
                        'file_extension': chunk.get('file_extension', ''),
                        'metadata': chunk.get('metadata', {})
                    }
                )
                points.append(point)
            
            # Загружаем батч в Qdrant Cloud
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Ждем подтверждения
                )
            except Exception as e:
                logger.error(f"Ошибка при загрузке батча {i//batch_size}: {e}")
                # Можно добавить логику повторной попытки
    
    def run_indexing(
        self,
        docs_folder: str = "docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        recreate_collection: bool = False,
        chunk_by: str = "characters"
    ):
        """
        Полный процесс индексации документов в Qdrant Cloud
        
        Args:
            docs_folder: Папка с документами
            chunk_size: Размер чанка
            chunk_overlap: Перекрытие чанков
            recreate_collection: Пересоздать коллекцию если существует
            chunk_by: Метод разбивки (characters, sentences, paragraphs)
        """
        logger.info("=" * 50)
        logger.info("НАЧАЛО ПРОЦЕССА ИНДЕКСАЦИИ")
        logger.info("=" * 50)
        
        # Проверка подключения
        if not self.check_connection():
            logger.error("Не удалось подключиться к Qdrant Cloud")
            return
        
        # 1. Чтение документов
        logger.info("1. Чтение документов...")
        documents = self.read_documents(docs_folder)
        
        if not documents:
            logger.warning("Нет документов для индексации")
            return
        
        # 2. Создание коллекции
        logger.info("2. Создание коллекции в Qdrant Cloud...")
        self.create_collection(recreate_if_exists=recreate_collection)
        
        # 3. Разбивка на чанки
        logger.info("3. Разбивка документов на чанки...")
        chunks = self.chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            chunk_by=chunk_by
        )
        
        # 4. Создание эмбеддингов
        logger.info("4. Создание эмбеддингов...")
        chunks_with_embeddings = self.create_embeddings(chunks)
        
        # 5. Индексация в Qdrant Cloud
        logger.info("5. Индексация в Qdrant Cloud...")
        self.index_to_qdrant(chunks_with_embeddings)
        
        # 6. Проверка результатов
        logger.info("6. Проверка результатов...")
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"Коллекция '{self.collection_name}' содержит {collection_info.points_count} точек")
        
        logger.info("=" * 50)
        logger.info("ИНДЕКСАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        logger.info("=" * 50)


# Пример использования
if __name__ == "__main__":
    # Конфигурация для Qdrant Cloud
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")  # Ваш URL из Qdrant Cloud
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Ваш API ключ
    
    # Инициализация индексатора
    indexer = QdrantCloudIndexer(
        cloud_url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
        collection_name="my_documents",
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Запуск индексации
    indexer.run_indexing(
        docs_folder="docs",  # Папка с документами
        chunk_size=1000,      # Размер чанка
        chunk_overlap=200,    # Перекрытие чанков
        recreate_collection=False,  # Пересоздать коллекцию если существует
        chunk_by="characters"  # Метод разбивки
    )