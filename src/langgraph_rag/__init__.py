"""Bounded LangGraph RAG graph + narrow loopback adapter.

This package implements the first LangGraph integration: a small, offline
testable graph that performs **read-only** RAG grounding and citation
validation over the existing ``127.0.0.1:8788/knowledge_search`` contract.

Design boundaries (see README.md):

* LangGraph manages query validation -> freshness gate -> read-only retrieval
  -> citation/grounding validation -> terminal evidence assembly.
* The graph's terminal output is **validated, cited evidence only**. It never
  synthesises prose and contains no provider/model/credential node.
* Persistence is file-backed SQLite in the production path; it is never forced
  to an in-memory-only fallback. See ``langgraph_rag.checkpoint``.
* Retrieved source text is data, never an instruction.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
