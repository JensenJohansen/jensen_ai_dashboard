# grok.py
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from core.grok_model import XAI_Grok

class GrokVanna(ChromaDB_VectorStore, XAI_Grok):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        XAI_Grok.__init__(self, config=config)