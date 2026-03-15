"""
Basic RAG (Retrieval-Augmented Generation) Application

How it works:
1. INGEST  — Load .txt files → split into chunks → embed → store in ChromaDB
2. RETRIEVE — Embed the user's question → find similar chunks in the DB
3. GENERATE — Send retrieved chunks + question to Claude → get a grounded answer
"""

import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from .env file

# testing PR review
class RAG:
    def __init__(
        self,
        docs_dir: str = "documents",
        collection_name: str = "my_docs",
        db_path: str = ".chroma_db",
    ):
        # Claude client for generation (reads ANTHROPIC_API_KEY from env)
        self.claude = anthropic.Anthropic()

        # ChromaDB stores vectors on disk so they persist between runs
        self.chroma = chromadb.PersistentClient(path=db_path)

        # Default embedding function uses all-MiniLM-L6-v2 (downloads on first run)
        # It converts text → 384-dimensional vectors that capture semantic meaning
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        # A collection is like a table — it holds your chunks and their embeddings
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

        self.docs_dir = docs_dir

    # ── Step 1: Ingestion ──────────────────────────────────────────────────────

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Split a long text into smaller overlapping chunks.

        Why overlap? So that a sentence split across two chunks still appears
        in full in at least one of them, preserving context.

        chunk_size=500 chars is a reasonable default for most prose documents.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap  # step forward, keeping `overlap` chars
        return [c for c in chunks if c]  # drop empty strings

    def ingest(self, force: bool = False):
        """
        Load every .txt file in docs_dir, chunk it, and store in ChromaDB.

        Uses upsert() so re-running this is safe — no duplicate embeddings.
        Set force=True to wipe the collection and re-index from scratch.
        """
        if force:
            # Delete and recreate the collection
            self.chroma.delete_collection(self.collection.name)
            self.collection = self.chroma.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embed_fn,
            )
            print("Collection cleared.")

        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"Created '{self.docs_dir}/' — add .txt files there and run again.")
            return

        txt_files = [f for f in os.listdir(self.docs_dir) if f.endswith(".txt")]
        if not txt_files:
            print(f"No .txt files found in '{self.docs_dir}/'.")
            return

        total_chunks = 0
        for filename in sorted(txt_files):
            path = os.path.join(self.docs_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self.chunk_text(text)

            # Each chunk needs a unique ID.  We use filename + position.
            ids = [f"{filename}::chunk::{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

            # upsert = insert or update (idempotent)
            # ChromaDB auto-generates embeddings via self.embed_fn
            self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

            print(f"  ✓ {filename}  ({len(chunks)} chunks)")
            total_chunks += len(chunks)

        print(f"\nIndexed {total_chunks} chunks from {len(txt_files)} file(s).")
        print("The vector database is saved in '.chroma_db/' for future runs.\n")

    # ── Step 2: Retrieval ──────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most semantically similar chunks to the query.

        ChromaDB embeds the query with the same model, then does a
        nearest-neighbor search using cosine similarity.
        """
        count = self.collection.count()
        if count == 0:
            return []

        # Don't ask for more results than we have
        n = min(top_k, count)
        results = self.collection.query(query_texts=[query], n_results=n)

        chunks = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "similarity": round(1 - distance, 3),  # distance → similarity score
            })
        return chunks

    # ── Step 3: Generation ─────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = 3, show_sources: bool = True) -> str:
        """
        Full RAG pipeline: retrieve relevant chunks, then ask Claude.

        The system prompt instructs Claude to answer *only* from the context,
        which keeps answers grounded and reduces hallucinations.
        """
        # --- Retrieve ---
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            return (
                "No documents are indexed yet. "
                "Call rag.ingest() first to load your documents."
            )

        if show_sources:
            print("\n[Retrieved chunks]")
            for i, c in enumerate(chunks, 1):
                print(f"  {i}. {c['source']}  (similarity: {c['similarity']})")
            print()

        # --- Build context ---
        context_parts = [
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        ]
        context = "\n\n---\n\n".join(context_parts)

        # --- Generate ---
        response = self.claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=(
                "You are a helpful assistant. "
                "Answer the user's question using ONLY the context provided below. "
                "If the answer is not contained in the context, say "
                "'I don't have enough information to answer that from the provided documents.' "
                "Cite the source filename when relevant."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )

        return response.content[0].text

    def stats(self):
        """Show how many chunks are currently indexed."""
        count = self.collection.count()
        print(f"Vector store contains {count} chunk(s).")
