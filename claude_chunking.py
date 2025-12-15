"""
Умный индексатор v3 для НПА
ИСПРАВЛЕНО: правильное чтение структуры docx
"""

import os
import re
import uuid
from typing import List, Optional, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartNPAIndexerV3:
    
    def __init__(
        self,
        cloud_url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "my_documents",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        cloud_url = cloud_url or os.getenv("QDRANT_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not cloud_url or not api_key:
            raise ValueError("Необходимо указать QDRANT_URL и QDRANT_API_KEY")
        
        self.client = QdrantClient(url=cloud_url, api_key=api_key, timeout=60)
        self.collection_name = collection_name
        
        logger.info(f"Загрузка модели: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
    
    def _read_docx_with_structure(self, file_path: str) -> List[Dict]:
        """
        Читаем DOCX с сохранением структуры абзацев
        Возвращаем список абзацев с их стилями
        """
        try:
            import docx
            doc = docx.Document(file_path)
            
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                # Определяем тип абзаца по стилю или содержимому
                style_name = para.style.name if para.style else ""
                
                paragraphs.append({
                    'text': text,
                    'style': style_name
                })
            
            return paragraphs
            
        except Exception as e:
            logger.error(f"Ошибка чтения docx {file_path}: {e}")
            return []
    
    def _parse_npa_structure(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        Парсим структуру НПА и создаём чанки
        """
        chunks = []
        
        current_context = {
            'chapter': '',
            'paragraph': '',  # Параграф НПА (не путать с абзацем)
        }
        
        current_chunk_texts = []
        current_points = []
        
        # Паттерны для распознавания структуры
        chapter_pattern = re.compile(r'^Глава\s+\d+', re.IGNORECASE)
        paragraph_pattern = re.compile(r'^Параграф\s+\d+', re.IGNORECASE)
        point_pattern = re.compile(r'^(\d+)\.\s+')
        
        for para in paragraphs:
            text = para['text']
            
            # Проверяем: это Глава?
            if chapter_pattern.match(text):
                # Сохраняем предыдущий чанк
                if current_chunk_texts:
                    chunks.append(self._create_chunk(
                        current_chunk_texts, 
                        current_context.copy(), 
                        current_points.copy()
                    ))
                    current_chunk_texts = []
                    current_points = []
                
                current_context['chapter'] = text
                current_context['paragraph'] = ''
                continue
            
            # Проверяем: это Параграф?
            if paragraph_pattern.match(text):
                # Сохраняем предыдущий чанк
                if current_chunk_texts:
                    chunks.append(self._create_chunk(
                        current_chunk_texts, 
                        current_context.copy(), 
                        current_points.copy()
                    ))
                    current_chunk_texts = []
                    current_points = []
                
                current_context['paragraph'] = text
                continue
            
            # Проверяем: это нумерованный пункт?
            point_match = point_pattern.match(text)
            if point_match:
                point_num = point_match.group(1)
                
                # Если накопили много текста — создаём чанк
                total_len = sum(len(t) for t in current_chunk_texts)
                if total_len > 800 and current_chunk_texts:
                    chunks.append(self._create_chunk(
                        current_chunk_texts, 
                        current_context.copy(), 
                        current_points.copy()
                    ))
                    current_chunk_texts = []
                    current_points = []
                
                current_points.append(point_num)
            
            current_chunk_texts.append(text)
        
        # Последний чанк
        if current_chunk_texts:
            chunks.append(self._create_chunk(
                current_chunk_texts, 
                current_context.copy(), 
                current_points.copy()
            ))
        
        return chunks
    
    def _create_chunk(self, texts: List[str], context: Dict, points: List[str]) -> Dict:
        """Создание чанка с контекстом"""
        content = "\n".join(texts)
        
        # Генерируем поисковые фразы на основе контекста
        search_phrases = self._generate_search_phrases(content, context)
        
        # Создаём текст для эмбеддинга (КОРОТКИЙ!)
        search_text = self._build_search_text(context, search_phrases, content[:200])
        
        # Текст для отображения
        display_text = self._build_display_text(content, context, points)
        
        return {
            'content': content,
            'display_text': display_text,
            'search_text': search_text,
            'search_phrases': search_phrases,
            'context': context,
            'points': points
        }
    
    def _generate_search_phrases(self, content: str, context: Dict) -> List[str]:
        """Генерация поисковых фраз"""
        phrases = []
        
        paragraph = context.get('paragraph', '').lower()
        content_lower = content.lower()
        
        # Анализируем заголовок параграфа
        if 'порядок' in paragraph:
            if 'субсидирован' in paragraph:
                phrases.extend([
                    "как получить субсидию",
                    "получение субсидии", 
                    "порядок получения субсидии",
                    "процедура субсидирования",
                    "этапы получения субсидии"
                ])
            if 'гарантирован' in paragraph:
                phrases.extend([
                    "как получить гарантию",
                    "получение гарантии",
                    "порядок гарантирования"
                ])
        
        if 'условия' in paragraph:
            phrases.extend([
                "условия получения",
                "требования для получения",
                "кто может получить"
            ])
        
        # Анализируем контент
        if 'обращается' in content_lower and 'заявк' in content_lower:
            phrases.extend(["подача заявки", "куда обращаться", "как подать заявку"])
        
        if 'отказ' in content_lower:
            phrases.extend(["отказ в субсидии", "причины отказа", "почему отказали"])
        
        if 'документ' in content_lower:
            phrases.extend(["какие документы нужны", "список документов", "пакет документов"])
        
        if 'срок' in content_lower:
            phrases.extend(["сроки рассмотрения", "сколько ждать"])
        
        return phrases
    
    def _build_search_text(self, context: Dict, phrases: List[str], content_start: str) -> str:
        """
        Текст для создания эмбеддинга
        ВАЖНО: должен быть коротким и релевантным!
        """
        parts = []
        
        # Заголовок параграфа (самое важное!)
        if context.get('paragraph'):
            parts.append(context['paragraph'])
        
        # Поисковые фразы
        if phrases:
            parts.append("Вопросы: " + ", ".join(phrases[:5]))
        
        # Начало контента
        parts.append(content_start)
        
        return " | ".join(parts)
    
    def _build_display_text(self, content: str, context: Dict, points: List[str]) -> str:
        """Текст для показа пользователю"""
        parts = []
        
        if context.get('chapter'):
            parts.append(context['chapter'])
        if context.get('paragraph'):
            parts.append(context['paragraph'])
        if points:
            if len(points) == 1:
                parts.append(f"Пункт {points[0]}")
            else:
                parts.append(f"Пункты {points[0]}-{points[-1]}")
        
        parts.append("---")
        parts.append(content)
        
        return "\n".join(parts)
    
    def process_documents(self, docs_folder: str) -> List[Dict]:
        """Обработка всех документов"""
        all_chunks = []
        
        files = [f for f in os.listdir(docs_folder) if f.endswith('.docx')]
        logger.info(f"Найдено {len(files)} документов")
        
        for filename in tqdm(files, desc="Обработка документов"):
            file_path = os.path.join(docs_folder, filename)
            
            # Читаем документ с сохранением структуры
            paragraphs = self._read_docx_with_structure(file_path)
            
            if not paragraphs:
                logger.warning(f"Пустой документ: {filename}")
                continue
            
            logger.info(f"  {filename}: {len(paragraphs)} абзацев")
            
            # Парсим структуру и создаём чанки
            doc_chunks = self._parse_npa_structure(paragraphs)
            
            logger.info(f"  → создано {len(doc_chunks)} чанков")
            
            # Добавляем метаданные
            doc_id = str(uuid.uuid4())
            for i, chunk in enumerate(doc_chunks):
                chunk['id'] = str(uuid.uuid4())
                chunk['document_id'] = doc_id
                chunk['filename'] = filename
                chunk['chunk_index'] = i
                chunk['total_chunks'] = len(doc_chunks)
                all_chunks.append(chunk)
        
        logger.info(f"\nВсего создано {len(all_chunks)} чанков")
        return all_chunks
    
    def create_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """Создание эмбеддингов из search_text"""
        logger.info("Создание эмбеддингов...")
        
        texts = [chunk['search_text'] for chunk in chunks]
        
        # Показываем примеры
        logger.info("\nПримеры search_text:")
        for text in texts[:5]:
            logger.info(f"  → {text[:100]}...")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Эмбеддинги"):
            batch = texts[i:i + batch_size]
            batch_emb = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.extend(batch_emb)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks
    
    def create_collection(self, recreate: bool = True):
        """Создание коллекции"""
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if exists and recreate:
            logger.info(f"Удаление коллекции '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
        elif exists:
            return
        
        logger.info(f"Создание коллекции '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
    
    def index_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """Загрузка в Qdrant"""
        logger.info(f"Загрузка {len(chunks)} чанков...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Загрузка"):
            batch = chunks[i:i + batch_size]
            
            points = []
            for chunk in batch:
                point = models.PointStruct(
                    id=chunk['id'],
                    vector=chunk['embedding'],
                    payload={
                        'content': chunk['display_text'],
                        'search_text': chunk['search_text'],
                        'search_phrases': chunk['search_phrases'],
                        'filename': chunk['filename'],
                        'document_id': chunk['document_id'],
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks'],
                        'chapter': chunk['context'].get('chapter', ''),
                        'paragraph': chunk['context'].get('paragraph', ''),
                        'points': chunk.get('points', [])
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
    
    def run_indexing(self, docs_folder: str = "docs", recreate_collection: bool = True):
        """Полный процесс"""
        logger.info("=" * 60)
        logger.info("ИНДЕКСАЦИЯ НПА v3")
        logger.info("=" * 60)
        
        self.create_collection(recreate=recreate_collection)
        
        chunks = self.process_documents(docs_folder)
        
        if not chunks:
            logger.error("Нет чанков для индексации!")
            return
        
        # Показываем статистику по параграфам
        paragraphs_found = [c for c in chunks if c['context'].get('paragraph')]
        logger.info(f"\nЧанков с параграфами: {len(paragraphs_found)}")
        for c in paragraphs_found[:10]:
            logger.info(f"  • {c['context']['paragraph'][:60]}...")
        
        chunks = self.create_embeddings(chunks)
        self.index_chunks(chunks)
        
        info = self.client.get_collection(self.collection_name)
        logger.info(f"\n✅ Готово! Точек в коллекции: {info.points_count}")


if __name__ == "__main__":
    indexer = SmartNPAIndexerV3(collection_name="my_documents")
    indexer.run_indexing(docs_folder="docs", recreate_collection=True)