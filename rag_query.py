#!/usr/bin/env python3
"""
Query the Qdrant vector database and return raw results.
Usage: python rag_query.py "your search query" [--top 5] [--category finance]
"""

import os, sys, json, argparse
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from embed import embed

COLLECTION  = os.getenv("QDRANT_COLLECTION", "agent_knowledge")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))


def query_rag(query_text: str, top_k: int = 5, category: str = None) -> list[dict]:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    except Exception as e:
        sys.exit(f"❌  Не удалось подключиться к Qdrant ({QDRANT_HOST}:{QDRANT_PORT}): {e}")

    try:
        vector = embed([query_text])[0]
    except Exception as e:
        sys.exit(f"❌  Ошибка при получении эмбеддинга: {e}")

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    try:
        # query_points — актуальный метод начиная с qdrant-client 1.7+
        # (client.search() объявлен устаревшим и будет удалён в следующих версиях)
        results = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        ).points
    except Exception as e:
        sys.exit(f"❌  Ошибка при запросе к Qdrant: {e}")

    return [
        {
            "score":    round(r.score, 4),
            "text":     r.payload.get("text", ""),
            "category": r.payload.get("category", ""),
            # md_to_qdrant.py сохраняет путь в поле "file";
            # поддерживаем также устаревшие поля "filename" / "source"
            # для совместимости с другими индексаторами
            "filename": r.payload.get("file") or r.payload.get("filename", ""),
            "source":   r.payload.get("source", ""),
        }
        for r in results
    ]


def main():
    parser = argparse.ArgumentParser(description="Query RAG knowledge base")
    parser.add_argument("query",      nargs="+",           help="Search query")
    parser.add_argument("--top",      type=int, default=5, help="Number of results")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (must match value stored in payload)")
    parser.add_argument("--json",     action="store_true", help="Output as JSON")
    args = parser.parse_args()

    query_text = " ".join(args.query)
    results    = query_rag(query_text, top_k=args.top, category=args.category)

    if args.json or not sys.stdout.isatty():
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        category_label = f" [category={args.category}]" if args.category else ""
        print(f"\n🔍  Query: '{query_text}'{category_label}\n{'-' * 60}")
        if not results:
            print("  (no results found)")
        for i, r in enumerate(results, 1):
            filename = r["filename"] or r["source"] or "—"
            category = r["category"] or "—"
            print(f"\n[{i}] Score: {r['score']} | {filename} (category: {category})")
            print(f"    {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
        print(f"\n{'-' * 60}")


if __name__ == "__main__":
    main()
