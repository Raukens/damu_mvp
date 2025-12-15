"""
RAG Service v2 - с Query Expansion через LLM
Улучшенный поиск за счёт переформулирования запросов
"""

import os
import json
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска"""
    content: str
    filename: str
    score: float
    chapter: str
    paragraph: str
    points: List[str]
    chunk_id: str = ""


@dataclass
class RAGResponse:
    """Ответ RAG системы"""
    answer: str
    sources: List[SearchResult]
    query: str
    expanded_queries: List[str] = field(default_factory=list)


class RAGServiceV2:
    """
    RAG сервис v2 с Query Expansion
    """
    
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        collection_name: str = "my_documents",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gpt-4o-mini"
    ):
        # Qdrant
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        # OpenAI
        self.openai_api_key = openai_api_key or os.getenv("GPT_KEY")
        self.llm_model = llm_model
        
        self._init_clients()
        
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info("RAG сервис v2 инициализирован")
    
    def _init_clients(self):
        """Инициализация клиентов"""
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("Необходимо указать QDRANT_URL и QDRANT_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("Необходимо указать OPENAI_API_KEY")
        
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=30
        )
        
        self.openai = OpenAI(api_key=self.openai_api_key)
    
    def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Расширение запроса через LLM
        Генерирует варианты запроса для лучшего поиска
        """
        logger.info(f"Расширение запроса: '{query}'")
        
        prompt = f"""Ты помогаешь улучшить поиск по нормативно-правовым актам Республики Казахстан.

Пользователь задал вопрос: "{query}"

Сгенерируй {num_variants} альтернативных формулировки этого запроса для поиска в юридических документах.

ПРАВИЛА:
1. Используй юридическую терминологию НПА РК (порядок, условия, требования, предоставление)
2. Каждый вариант должен искать информацию под разным углом
3. Включи ключевые слова которые могут быть в заголовках разделов
4. Один вариант — разговорный, один — формальный юридический

Верни ТОЛЬКО JSON массив строк, без пояснений:
["вариант 1", "вариант 2", "вариант 3"]"""

        try:
            response = self.openai.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Парсим JSON
            # Убираем возможные markdown блоки
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            variants = json.loads(result_text)
            
            # Добавляем оригинальный запрос в начало
            all_queries = [query] + variants
            
            logger.info(f"Сгенерировано запросов: {len(all_queries)}")
            for i, q in enumerate(all_queries):
                logger.info(f"  {i+1}. {q}")
            
            return all_queries
            
        except Exception as e:
            logger.warning(f"Ошибка расширения запроса: {e}")
            # Fallback: возвращаем только оригинальный запрос
            return [query]
    
    def search_single(self, query: str, limit: int = 5, score_threshold: float = 0.3) -> List[SearchResult]:
        """Поиск по одному запросу"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        results = []
        for hit in search_result.points:
            payload = hit.payload
            result = SearchResult(
                content=payload.get('content', ''),
                filename=payload.get('filename', ''),
                score=hit.score,
                chapter=payload.get('chapter', ''),
                paragraph=payload.get('paragraph', ''),
                points=payload.get('points', []),
                chunk_id=str(hit.id)
            )
            results.append(result)
        
        return results
    
    def search_with_expansion(
        self, 
        query: str, 
        limit: int = 5, 
        score_threshold: float = 0.3,
        expand: bool = True,
        num_variants: int = 3
    ) -> tuple[List[SearchResult], List[str]]:
        """
        Поиск с расширением запроса
        Возвращает объединённые результаты и список использованных запросов
        """
        if expand:
            queries = self.expand_query(query, num_variants)
        else:
            queries = [query]
        
        # Собираем результаты от всех запросов
        all_results: Dict[str, SearchResult] = {}  # chunk_id -> result
        
        for q in queries:
            results = self.search_single(q, limit=limit, score_threshold=score_threshold)
            
            for result in results:
                chunk_id = result.chunk_id
                
                # Если чанк уже найден — берём лучший score
                if chunk_id in all_results:
                    if result.score > all_results[chunk_id].score:
                        all_results[chunk_id] = result
                else:
                    all_results[chunk_id] = result
        
        # Сортируем по score
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        # Ограничиваем количество
        final_results = sorted_results[:limit]
        
        logger.info(f"Найдено уникальных чанков: {len(all_results)}, возвращаем топ-{len(final_results)}")
        
        return final_results, queries
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Формирование контекста"""
        if not search_results:
            return "Релевантная информация не найдена."
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            source_info = f"[Источник {i} | Релевантность: {result.score:.0%}]"
            if result.filename:
                source_info += f"\nДокумент: {result.filename}"
            if result.chapter:
                source_info += f"\n{result.chapter}"
            if result.paragraph:
                source_info += f"\n{result.paragraph}"
            if result.points:
                source_info += f"\nПункты: {', '.join(result.points)}"
            
            context_parts.append(f"{source_info}\n\n{result.content}")
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, expanded_queries: List[str]) -> str:
        """Формирование промпта"""
        
        # Показываем какие запросы использовались
        queries_info = ""
        if len(expanded_queries) > 1:
            queries_info = f"""
Для поиска информации использовались следующие варианты запроса:
{chr(10).join(f'- {q}' for q in expanded_queries)}
"""
        
        return f"""Ты — эксперт-консультант по нормативно-правовым актам Республики Казахстан.
Твоя задача — давать точные, полные и практически полезные ответы.

ПРАВИЛА ОТВЕТА:
1. Отвечай СТРОГО на основе предоставленного контекста
2. Структурируй ответ: краткий ответ → детали → пошаговые действия (если применимо)
3. ОБЯЗАТЕЛЬНО указывай номера пунктов при цитировании (например: "согласно п. 23...")
4. Если информации недостаточно — честно скажи об этом
5. Используй понятный язык, но сохраняй юридическую точность
{queries_info}
КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

Дай структурированный ответ:"""
    
    def generate_answer(
        self, 
        query: str, 
        search_results: List[SearchResult],
        expanded_queries: List[str],
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """Генерация ответа"""
        context = self._build_context(search_results)
        prompt = self._build_prompt(query, context, expanded_queries)
        
        logger.info(f"Генерация ответа через {self.llm_model}")
        
        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def ask(
        self, 
        query: str, 
        num_chunks: int = 5,
        score_threshold: float = 0.3,
        temperature: float = 0.3,
        expand_query: bool = True,
        num_variants: int = 3
    ) -> RAGResponse:
        """
        Основной метод с Query Expansion
        
        Args:
            query: Вопрос пользователя
            num_chunks: Количество чанков контекста
            score_threshold: Минимальный порог релевантности
            temperature: Температура генерации
            expand_query: Использовать ли расширение запроса
            num_variants: Количество вариантов запроса
        """
        # 1. Поиск с расширением
        search_results, expanded_queries = self.search_with_expansion(
            query=query,
            limit=num_chunks,
            score_threshold=score_threshold,
            expand=expand_query,
            num_variants=num_variants
        )
        
        # 2. Генерация ответа
        if search_results:
            answer = self.generate_answer(
                query=query,
                search_results=search_results,
                expanded_queries=expanded_queries,
                temperature=temperature
            )
        else:
            answer = "К сожалению, я не нашёл релевантной информации в документах. Попробуйте переформулировать вопрос."
        
        return RAGResponse(
            answer=answer,
            sources=search_results,
            query=query,
            expanded_queries=expanded_queries
        )


def compare_modes():
    """
    Сравнение режимов: с Query Expansion и без
    """
    print("=" * 70)
    print("🔬 СРАВНЕНИЕ: Query Expansion vs Обычный поиск")
    print("=" * 70)
    
    rag = RAGServiceV2()
    
    test_query = "Как получить субсидии?"
    
    # Без расширения
    print("\n📌 БЕЗ Query Expansion:")
    print("-" * 50)
    results_no_expand, queries_no = rag.search_with_expansion(
        test_query, limit=5, expand=False
    )
    print(f"Запросы: {queries_no}")
    for i, r in enumerate(results_no_expand, 1):
        print(f"  {i}. Score: {r.score:.3f} | {r.paragraph[:50]}...")
    
    # С расширением
    print("\n📌 С Query Expansion:")
    print("-" * 50)
    results_expand, queries_yes = rag.search_with_expansion(
        test_query, limit=5, expand=True, num_variants=3
    )
    print(f"Запросы: {queries_yes}")
    for i, r in enumerate(results_expand, 1):
        print(f"  {i}. Score: {r.score:.3f} | {r.paragraph[:50]}...")
    
    # Сравнение
    print("\n" + "=" * 70)
    print("📊 ИТОГ:")
    
    best_no = results_no_expand[0].score if results_no_expand else 0
    best_yes = results_expand[0].score if results_expand else 0
    
    print(f"  Лучший score без расширения: {best_no:.3f}")
    print(f"  Лучший score с расширением:  {best_yes:.3f}")
    print(f"  Улучшение: +{(best_yes - best_no):.3f} ({((best_yes/best_no - 1) * 100):.1f}%)")


def interactive_mode():
    """Интерактивный режим"""
    print("=" * 60)
    print("🤖 RAG Консультант v2 (с Query Expansion)")
    print("=" * 60)
    print("Команды:")
    print("  'exit'    — выход")
    print("  'sources' — показать источники")
    print("  'simple'  — режим без Query Expansion")
    print("  'expand'  — режим с Query Expansion (по умолчанию)")
    print("  'compare' — сравнить режимы")
    print("-" * 60)
    
    try:
        rag = RAGServiceV2()
        print("✅ Сервис запущен\n")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return
    
    last_response = None
    use_expansion = True
    
    while True:
        mode_indicator = "🔄" if use_expansion else "📝"
        query = input(f"\n{mode_indicator} Ваш вопрос: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break
        
        if query.lower() == 'simple':
            use_expansion = False
            print("✅ Режим: простой поиск (без Query Expansion)")
            continue
        
        if query.lower() == 'expand':
            use_expansion = True
            print("✅ Режим: с Query Expansion")
            continue
        
        if query.lower() == 'compare':
            compare_modes()
            continue
        
        if query.lower() == 'sources' and last_response:
            print("\n📚 Источники:")
            for i, source in enumerate(last_response.sources, 1):
                print(f"\n[{i}] Score: {source.score:.3f}")
                print(f"    {source.chapter}")
                print(f"    {source.paragraph}")
                print(f"    Пункты: {source.points}")
            
            if last_response.expanded_queries:
                print("\n🔄 Использованные запросы:")
                for q in last_response.expanded_queries:
                    print(f"    • {q}")
            continue
        
        print("\n⏳ Обрабатываю запрос...")
        
        try:
            response = rag.ask(
                query, 
                expand_query=use_expansion,
                num_variants=3
            )
            last_response = response
            
            print(f"\n💡 Ответ:\n")
            print(response.answer)
            
            print(f"\n📎 Источников: {len(response.sources)} | ", end="")
            print(f"Запросов: {len(response.expanded_queries)}")
            print("   (введите 'sources' для деталей)")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_modes()
    else:
        interactive_mode()