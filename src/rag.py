"""
RAG pipeline for HR policy documents.

Flow:  Document -> Chunk -> Embed -> FAISS -> Retrieve

Embeddings are generated locally with a small sentence-transformers model, so
no embedding API key is required. FAISS holds the vectors in memory for the
session.
"""
from __future__ import annotations

import logging
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


def load_embeddings() -> HuggingFaceEmbeddings:
    """Create the local embedding model (cache this in the app layer)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class RAGPipeline:
    """Chunk, embed, and retrieve HR policy documents using FAISS."""

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embeddings = embeddings
        self.vector_store: FAISS | None = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None

    # --- Ingestion ------------------------------------------------------------
    def index_directory(self, directory: str) -> int:
        """Load all .txt policy files from a directory and index them.

        Returns the number of source documents indexed.
        """
        if not os.path.isdir(directory):
            logger.warning("Policies directory not found: %s", directory)
            return 0

        documents: list[Document] = []
        for filename in sorted(os.listdir(directory)):
            if not filename.lower().endswith(".txt"):
                continue
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents.append(
                    Document(page_content=f.read(), metadata={"source": filename})
                )

        if documents:
            self._add(documents)
        return len(documents)

    def index_uploaded_file(self, uploaded_file) -> bool:
        """Index a Streamlit UploadedFile (.txt or .pdf). Returns success."""
        name = uploaded_file.name
        if name.lower().endswith(".txt"):
            content = uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.lower().endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(uploaded_file)
            content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            logger.warning("Unsupported file type: %s", name)
            return False

        if not content.strip():
            logger.warning("No extractable text in %s", name)
            return False

        self._add([Document(page_content=content, metadata={"source": name})])
        return True

    def _add(self, documents: list[Document]) -> None:
        """Chunk documents and add them to the FAISS store."""
        chunks = self.splitter.split_documents(documents)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        logger.info("Indexed %d chunks from %d document(s)", len(chunks), len(documents))

    # --- Retrieval ------------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Return the top-k relevant chunks as {content, source} dicts."""
        if self.vector_store is None:
            return []
        k = k or config.RETRIEVAL_K
        docs = self.vector_store.similarity_search(query, k=k)
        return [
            {"content": d.page_content, "source": d.metadata.get("source", "Unknown")}
            for d in docs
        ]

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        if not chunks:
            return ""
        return "\n\n---\n\n".join(
            f"[{c['source']}]\n{c['content']}" for c in chunks
        )

    @staticmethod
    def sources(chunks: list[dict]) -> list[str]:
        """Return the unique source filenames, order-preserved."""
        seen: list[str] = []
        for c in chunks:
            if c["source"] not in seen:
                seen.append(c["source"])
        return seen
