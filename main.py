"""
Interactive RAG demo — run this to try out your RAG application.

Usage:
    python main.py
"""

from rag import RAG


def main():
    print("=" * 55)
    print("       Basic RAG Application with Claude")
    print("=" * 55)

    rag = RAG(docs_dir="documents")

    # ── Ingest ──────────────────────────────────────────────────
    print("\n[Step 1] Indexing documents in 'documents/'...\n")
    rag.ingest()
    rag.stats()

    # ── Interactive Q&A loop ────────────────────────────────────
    print("[Step 2] Ask questions about your documents.")
    print("         Type 'quit' to exit, 'stats' for index info.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "stats":
            rag.stats()
            continue

        print("\nSearching knowledge base and generating answer...")
        answer = rag.ask(question)
        print(f"Answer:\n{answer}")
        print("\n" + "-" * 55 + "\n")


if __name__ == "__main__":
    main()
