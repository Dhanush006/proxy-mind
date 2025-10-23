import os
import shutil
import yaml
import faiss
import numpy as np
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import (
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    NotebookLoader,
    UnstructuredFileLoader
)

class IngestEngine:
    def __init__(self, data_dir=".", persist_dir_base="vector_dbs", 
                 config_path="config/chunking_config.yaml"):
        self.data_dir = data_dir
        self.persist_dir_chroma = os.path.join(persist_dir_base, "db_semantic")
        self.persist_dir_faiss = os.path.join(persist_dir_base, "db_precise")

        # Embeddings
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Configs
        self.chunk_config = self.load_config(config_path)

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_loader_by_extension(self, ext):
        return {
            ".py": PythonLoader,
            ".ipynb": NotebookLoader,
            ".txt": TextLoader,
            ".env": TextLoader,
            ".md": TextLoader,
            ".pdf": PyPDFLoader,
            ".mat": None,  # .mat not supported yet
        }.get(ext, TextLoader)

    def get_splitter_for_extension(self, ext, splitter=RecursiveCharacterTextSplitter):
        config = self.chunk_config.get(ext, {"chunk_size": 500, "chunk_overlap": 50})
        if splitter == RecursiveCharacterTextSplitter:
            return RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""]
        )
        else:
            return TokenTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
            )
        

    def load_all_documents(self) -> List:
        all_docs = []
        ignore_dirs = {"vector_dbs", ".git", "__pycache__"}

        for root, dirs, files in os.walk(self.data_dir):
            # Filter out ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                ext = Path(file).suffix
                loader_cls = self.get_loader_by_extension(ext)
                file_path = os.path.join(root, file)

                try:
                    loader = loader_cls(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"✅ Loaded {len(docs)} docs from {file}")
                except Exception as e:
                    print(f"⚠️ Skipping {file}: {e}")
        
        return all_docs


    def split_documents(self, docs: List[Document]) -> List[Document]:
        split_docs = []
        for doc in docs:
            ext = Path(doc.metadata.get("source", "")).suffix
            if ext in [".py", ".ipynb", ".m"]:
                splitter_type = TokenTextSplitter
            else:
                splitter_type = RecursiveCharacterTextSplitter
            splitter = self.get_splitter_for_extension(ext, splitter=splitter_type)
            split_docs.extend(splitter.split_documents([doc]))
        print(f"🧩 Total chunks: {len(split_docs)}")
        return split_docs

    def run_full_ingestion(self):
        raw_docs = self.load_all_documents()
        chunks = self.split_documents(raw_docs)

        if os.path.exists(self.persist_dir_chroma):
            shutil.rmtree(self.persist_dir_chroma)

        chroma_db = Chroma.from_documents(
            chunks, self.embedding,
            persist_directory=self.persist_dir_chroma
        )
        print(f"✅ ChromaDB saved at {self.persist_dir_chroma}")

        self.build_faiss_ivf_index(chunks)

    def build_faiss_ivf_index(self, chunks: List[Document]):
        """
        Builds a FAISS index using IVF + L2 distance with fallback to FlatL2 if vector count is too low.
        Saves the final index and vector store in `vector_dbs/db_precise`.
        """

        print("🔧 Generating FAISS IVF index...")

        # Step 1: Convert chunks into texts and their corresponding embeddings
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        vectors = self.embedding.embed_documents(texts)
        vectors_np = np.array(vectors).astype("float32")
        dimension = vectors_np.shape[1]
        num_vectors = vectors_np.shape[0]

        # Step 2: Dynamic nlist and nprobe
        sqrt_n = int(np.sqrt(num_vectors))
        nlist =  max(4, min(sqrt_n, 128))
        nprobe = max(1, min(nlist // 10, 20))

        print(f"📏 num_vectors={num_vectors} | dimension={dimension} | nlist={nlist} | nprobe={nprobe}")

        # Step 3: Create quantizer and IVF index
        quantizer = faiss.IndexFlatL2(dimension)
        use_ivf = num_vectors >= (nlist * 39)

        if use_ivf:
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            print("📈 Training IVF index...")
            try:
                index.train(vectors_np)
                index.nprobe = nprobe
                print("📦 Adding vectors to IVF index...")
                index.add(vectors_np)
            except Exception as e:
                print(f"⚠️ IVF training failed: {e}. Falling back to FlatL2 index.")
                index = faiss.IndexFlatL2(dimension)
                index.add(vectors_np)
        else:
            print(f"⚠️ Not enough vectors ({num_vectors}) to train IVF with nlist={nlist}. Falling back to FlatL2 index.")
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors_np)

        # Step 4: Wrap with FAISS Vectorstore
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(chunks)})
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

        faiss_db = FAISS(
            embedding_function=self.embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )

        # Step 5: Save vectorstore locally
        persist_path = "vector_dbs/db_precise"
        faiss_db.save_local(persist_path)
        print(f"✅ FAISS DB saved at {persist_path} using {'IVF' if use_ivf else 'FlatL2'} index.")




if __name__ == "__main__":
    engine = IngestEngine()
    engine.run_full_ingestion()
