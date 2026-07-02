"""Knowledge / RAG layer.

Ingests documents (PDF, Markdown, Word, PowerPoint, CSV, Excel, web pages, code,
directories), chunks and embeds them, and retrieves relevant context to ground the
agents' answers.
"""

from ai_employee.knowledge.chunking import chunk_text
from ai_employee.knowledge.loaders import LoadedDocument, load_source
from ai_employee.knowledge.rag import KnowledgeBase

__all__ = ["KnowledgeBase", "load_source", "LoadedDocument", "chunk_text"]
