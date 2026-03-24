"""
md_to_qdrant.py
---------------
Читает .md файл или папку с .md файлами, получает эмбеддинги через OpenAI или
OpenRouter и сохраняет их в Qdrant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ЗАВИСИМОСТИ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install openai>=1.30.0 qdrant-client>=1.9.0 python-dotenv>=1.0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ПРИМЕР .env ФАЙЛА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Провайдер: "openai" или "openrouter"
    EMBEDDING_PROVIDER=openai

    # OpenAI (если EMBEDDING_PROVIDER=openai)
    OPENAI_API_KEY=sk-...

    # OpenRouter (если EMBEDDING_PROVIDER=openrouter)
    OPENROUTER_API_KEY=sk-or-...

    # Модель эмбеддингов
    # OpenAI:     text-embedding-3-small (1536)
    #             text-embedding-3-large (3072)
    #             text-embedding-ada-002 (1536)
    # OpenRouter: openai/text-embedding-3-small
    EMBEDDING_MODEL=text-embedding-3-small

    # Размерность — должна совпадать с выбранной моделью
    EMBEDDING_DIM=1536

    # Qdrant
    QDRANT_URL=http://localhost:6333
    # QDRANT_API_KEY=your-key   # только для облачного Qdrant
    QDRANT_COLLECTION=md_embeddings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ИСПОЛЬЗОВАНИЕ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python md_to_qdrant.py <путь_к_файлу_или_папке> [опции]

Примеры:
    python md_to_qdrant.py ./docs/
    python md_to_qdrant.py README.md
    python md_to_qdrant.py ./docs/ --collection my_docs --chunk-size 500
    python md_to_qdrant.py ./docs/ --category tech_docs
    python md_to_qdrant.py ./docs/ --dry-run
"""

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ─────────────────────────────────────────────
# Загрузка переменных окружения
# ─────────────────────────────────────────────
load_dotenv()

PROVIDER          = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # "openai" | "openrouter"
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY= os.getenv("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "md_embeddings")

# Размерность модели по умолчанию (text-embedding-3-small = 1536)
EMBEDDING_DIM     = int(os.getenv("EMBEDDING_DIM", "1536"))


# ─────────────────────────────────────────────
# Клиент для эмбеддингов
# ─────────────────────────────────────────────

def build_embedding_client() -> OpenAI:
    """Создаёт OpenAI-совместимый клиент в зависимости от выбранного провайдера."""
    if PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            sys.exit("❌  OPENROUTER_API_KEY не задан в .env")
        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    else:  # openai (по умолчанию)
        if not OPENAI_API_KEY:
            sys.exit("❌  OPENAI_API_KEY не задан в .env")
        return OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Работа с Markdown-файлами
# ─────────────────────────────────────────────

def collect_md_files(path: Path) -> list[Path]:
    """Возвращает список .md файлов — одиночный файл или все файлы в папке."""
    if path.is_file():
        if path.suffix.lower() != ".md":
            sys.exit(f"❌  Файл {path} не является .md файлом")
        return [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.md"))
        if not files:
            sys.exit(f"❌  В папке {path} не найдено .md файлов")
        return files
    else:
        sys.exit(f"❌  Путь {path} не существует")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбивает текст на чанки по словам с перекрытием.
    Старается не разрывать абзацы.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # перекрытие: берём хвост из текущего буфера
            overlap_words = current[-overlap:] if overlap else []
            current = overlap_words + words
            current_len = len(current)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ─────────────────────────────────────────────
# Эмбеддинги
# ─────────────────────────────────────────────

def get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Запрашивает эмбеддинги батчем (до 2048 строк за раз)."""
    BATCH = 512
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        # убираем слишком длинные строки (>= 8191 токенов — лимит OpenAI)
        batch = [t[:32_000] for t in batch]
        response = client.embeddings.create(input=batch, model=model)
        all_vectors.extend([item.embedding for item in response.data])

    return all_vectors


# ─────────────────────────────────────────────
# Qdrant
# ─────────────────────────────────────────────

def ensure_collection(qdrant: QdrantClient, collection: str, dim: int) -> None:
    """Создаёт коллекцию, если она не существует."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"✅  Коллекция «{collection}» создана (dim={dim})")
    else:
        print(f"ℹ️   Коллекция «{collection}» уже существует")


def upsert_points(
    qdrant: QdrantClient,
    collection: str,
    vectors: list[list[float]],
    payloads: list[dict],
) -> None:
    """Загружает точки в Qdrant батчами по 100."""
    BATCH = 100
    for i in range(0, len(vectors), BATCH):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j],
                payload=payloads[j],
            )
            for j, _ in enumerate(vectors[i : i + BATCH], start=i)
        ]
        qdrant.upsert(collection_name=collection, points=points)


# ─────────────────────────────────────────────
# Основная логика
# ─────────────────────────────────────────────

def process_files(
    input_path: Path,
    collection: str,
    chunk_size: int,
    overlap: int,
    dry_run: bool,
    category: Optional[str] = None,
) -> None:
    md_files = collect_md_files(input_path)
    print(f"📄  Найдено файлов: {len(md_files)}")
    if category:
        print(f"🏷️   Категория: {category}")

    embed_client = build_embedding_client()
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not dry_run:
        ensure_collection(qdrant, collection, EMBEDDING_DIM)

    all_texts: list[str] = []
    all_payloads: list[dict] = []

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(content, chunk_size, overlap)

        print(f"  📝  {md_file}  →  {len(chunks)} чанков")

        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)

            payload = {
                "file": str(md_file),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "text": chunk,
            }
            # Добавляем category в payload только если она передана.
            # Это позволяет фильтровать точки в Qdrant по полю category,
            # например: Filter(must=[FieldCondition(key="category", match=MatchValue(value="..."))]).
            if category is not None:
                payload["category"] = category

            all_payloads.append(payload)

    if not all_texts:
        print("⚠️   Нет текста для обработки")
        return

    print(f"\n🔢  Получаем эмбеддинги для {len(all_texts)} чанков "
          f"(провайдер={PROVIDER}, модель={EMBEDDING_MODEL}) ...")

    vectors = get_embeddings(embed_client, all_texts, EMBEDDING_MODEL)

    if dry_run:
        print(f"\n🔍  [dry-run] Было бы сохранено {len(vectors)} векторов "
              f"в коллекцию «{collection}»"
              + (f" с категорией «{category}»" if category else ""))
        return

    print(f"💾  Сохраняем в Qdrant ({QDRANT_URL}, коллекция={collection}) ...")
    upsert_points(qdrant, collection, vectors, all_payloads)

    print(f"\n✅  Готово! Загружено {len(vectors)} векторов"
          + (f" с категорией «{category}»." if category else "."))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексирует .md файлы в Qdrant через OpenAI / OpenRouter эмбеддинги"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Путь к .md файлу или папке с .md файлами",
    )
    parser.add_argument(
        "--collection",
        default=QDRANT_COLLECTION,
        help=f"Название коллекции Qdrant (по умолчанию: {QDRANT_COLLECTION})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Максимальный размер чанка в словах (по умолчанию: 300)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Количество слов перекрытия между чанками (по умолчанию: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Не сохранять в Qdrant, только показать что было бы загружено",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Необязательная метка категории, которая сохраняется в payload каждого чанка. "
            "Позволяет фильтровать результаты RAG-поиска по категории. "
            "Пример: --category tech_docs"
        ),
    )

    args = parser.parse_args()
    process_files(
        input_path=args.path,
        collection=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        dry_run=args.dry_run,
        category=args.category,
    )


if __name__ == "__main__":
    main()
