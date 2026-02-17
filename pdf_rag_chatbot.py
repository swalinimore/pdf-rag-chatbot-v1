# imports
import ollama
import numpy as np
import faiss
import hashlib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Define class
class RAGEngine:
    # Define init to declare all the state variables
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.metadata = []
        self.doc_hashes = {}
        self.MAX_TURNS = 5
        self.chat_history = []

    # The flow:
    # 1. Upload phase
    # Files will be uploaded using streamlit UI. 
    # add_document file will pass those files to _extract_pdf_page function.
    # Questions - will you pass all files here or a single file?
    def _extract_pdf_page(self,pdf):
        pages = []
        full_text = []
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages, start = 1):
            text = page.extract_text()
            if text:
                pages.append({
                    "content": text.strip(),
                    "page_num": i,
                    "source":getattr(pdf,"name",f"pdf_{i}")
                    })
                full_text.append(text)
        return pages, "\n\n".join(full_text)
    
    def _compute_hash(self,pdf_text: str) -> str:
        return hashlib.sha256(pdf_text.encode("utf-8")).hexdigest()
    
    def _chunking(self,pages,chunk_size=800, overlap=200):
        chunks = []
        for page in pages:
            text = page["content"]
            #page_num = page["page_num"]
            start = 0 
            chunk_id = 1
            while start < len(text):
                end = start + chunk_size
                chunks.append({
                    "content":text[start:end],
                    "page_num": page["page_num"],
                    "source":page["source"],
                    "chunk_id" : f"{page['source']}_{page['page_num']}_{chunk_id}"
                })
                start = end - overlap
                chunk_id += 1
        return chunks

    # 2. Build Index phase

    def _build_index(self,chunks):
        texts = [c["content"] for c in chunks]
        embeddings = self.model.encode(texts,normalize_embeddings=True)
        vectors = embeddings.astype("float32")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadata = []
        for c in chunks:
            self.metadata.append(c)

    def add_document(self,pdf_files):
        all_chunks = []
        rebuild_index = False
        for pdf in pdf_files:
            pages, full_text = self._extract_pdf_page(pdf)
            doc_hash = self._compute_hash(full_text)
            if pdf.name not in self.doc_hashes or self.doc_hashes[pdf.name] != doc_hash:
                self.doc_hashes[pdf.name] = doc_hash
                rebuild_index = True
                chunks = self._chunking(pages)
                all_chunks.extend(chunks)
        if rebuild_index and all_chunks:
            self._build_index(all_chunks)    

    # 3. Query phase
    def retrieve(self,query,top_k=3,min_score=0.3):
        if self.index is None:
            return []
        
        q_vec = self.model.encode([query],normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_vec,top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if score < min_score:
                continue
            m = self.metadata[idx]
            results.append({**m, "score": float(score)})

        return results
    
    def _add_to_chat_history(self, role, content):
        self.chat_history.append({
            "role": role,
            "content": content
        })

        # keep only last N turns
        max_messages = self.MAX_TURNS * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    # 4. PromptBuilder
    def _promptbuilder(self, query):
        SYSTEM_PROMPT = """
                    You are a helpful assistant answering questions using ONLY the provided context.

                    Rules:
                    1. Use only the information from the context.
                    2. If the answer is not present, say: 
                    "I don't know based on the provided documents."
                    3. Do not use outside knowledge.
                    4. Be concise and factual.
                    5. Cite sources using (source: <filename>, page: <page number>)."""
        response_chunks = self.retrieve(query)
        if not response_chunks:
            return [
            {"role": "system", 
             "content": """You are a helpful assistant. 
             Your only response when there is no context provided is- 'I don't know based on the provided documents.'"""},
            {"role": "user", "content": " "}
        ]
        context = "\n\n".join(f"(source: {c['source']}, page_num: {c['page_num']})\n{c['content']}"
            for c in response_chunks
        )
        prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                #{"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
        prompt.extend(self.chat_history[:-1])
        prompt.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"})
        return prompt 

    # 5. Answer phase
    def answer(self, query):
        self._add_to_chat_history("user",query)
        prompt = self._promptbuilder(query)
        response = ollama.chat(
            model="llama3",
            messages=prompt
        )
        assistant_reply = response["message"]["content"]
        self._add_to_chat_history("assistant", assistant_reply)
        return assistant_reply
