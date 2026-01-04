# import chromadb
# from sentence_transformers import SentenceTransformer
# from PIL import Image
# from utils.logger import logger
# import uuid

# class VectorStore:
#     def __init__(self, persistent_path="./chroma_db"):
#         self.client = chromadb.PersistentClient(path=persistent_path)
        
#         # Text Embedding Model
#         logger.info("Loading Text Embedding Model (all-MiniLM-L6-v2)...")
#         self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
#         self.pdf_collection = self.client.get_or_create_collection(name="pdf_knowledge")

#         # Image Embedding Model (CLIP)
#         logger.info("Loading Image Embedding Model (clip-ViT-B-32)...")
#         self.clip_model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         self.image_collection = self.client.get_or_create_collection(name="image_gallery")

#     def add_pdf_content(self, content_list):
#         """
#         Adds PDF pages/chunks to the vector database.
#         content_list: list of dicts {text, page, source}
#         """
#         if not content_list:
#             return
        
#         documents = [item['text'] for item in content_list]
#         metadatas = [{"source": item['source'], "page": item['page']} for item in content_list]
#         ids = [str(uuid.uuid4()) for _ in content_list]
        
#         embeddings = self.text_model.encode(documents).tolist()
        
#         self.pdf_collection.add(
#             documents=documents,
#             embeddings=embeddings,
#             metadatas=metadatas,
#             ids=ids
#         )
#         logger.info(f"Added {len(documents)} text chunks to PDF collection.")

#     def add_image(self, image_path, image_obj):
#         """
#         Adds an image to the vector database.
#         """

#         try:
#             # CLIP Encode Image
#             embedding = self.clip_model.encode(image_obj).tolist()

#             self.image_collection.add(
#                 embeddings=[embedding],
#                 metadatas=[{"source": image_path}],
#                 ids=[str(uuid.uuid4())]
#             )

#             logger.info(f"Added image {image_path} to Image collection.")
#         except Exception as e:
#             logger.error(f"Failed to index image {image_path}: {e}")

#     def search_text(self, query, n_results=5):
#         """
#         Search for text in PDF collection.
#         """
#         query_embedding = self.text_model.encode([query]).tolist()
#         results = self.pdf_collection.query(
#             query_embeddings=query_embedding,
#             n_results=n_results
#         )
#         return results

#     def search_image(self, query_text, n_results=5):
#         """
#         Search for images using text description (Text-to-Image).
#         """

#         query_embedding = self.clip_model.encode([query_text]).tolist()
#         results = self.image_collection.query(
#             query_embeddings=query_embedding,
#             n_results=n_results
#         )
#         return results


import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
from utils.logger import logger
import uuid

class VectorStore:
    def __init__(self, persistent_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persistent_path)
        
        # Text Embedding Model
        logger.info("Loading Text Embedding Model (all-MiniLM-L6-v2)...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.pdf_collection = self.client.get_or_create_collection(name="pdf_knowledge")
        self.abstract_collection = self.client.get_or_create_collection(name="paper_abstracts")

        # Image Embedding Model (CLIP)
        logger.info("Loading Image Embedding Model (clip-ViT-B-32)...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32', device='cpu')
        self.image_collection = self.client.get_or_create_collection(name="image_gallery")

    def add_pdf_content(self, content_list):
        """
        Adds PDF pages/chunks to the vector database.
        content_list: list of dicts {text, page, source, abstract}
        """
        if not content_list:
            return
        
        # 1. Add all pages to full-text collection
        documents = [item['text'] for item in content_list]
        metadatas = [{"source": item['source'], "page": item['page']} for item in content_list]
        ids = [str(uuid.uuid4()) for _ in content_list]
        
        embeddings = self.text_model.encode(documents).tolist()
        
        self.pdf_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # 2. Add abstract to abstract collection if present (usually in the first item)
        # We index the abstract for the whole paper
        abstract_item = next((item for item in content_list if item.get('abstract')), None)
        if abstract_item:
            source = abstract_item['source']
            abstract_text = abstract_item['abstract']
            
            # Check if abstract already exists for this source to avoid duplicates
            existing = self.abstract_collection.get(where={"source": source})
            if not existing['ids']:
                abs_embedding = self.text_model.encode([abstract_text]).tolist()
                self.abstract_collection.add(
                    documents=[abstract_text],
                    embeddings=abs_embedding,
                    metadatas=[{"source": source}],
                    ids=[str(uuid.uuid4())]
                )
        
        logger.info(f"Added {len(documents)} text chunks to PDF collection.")

    def add_image(self, image_path, image_obj):
        # ... (same as before)
        try:
            embedding = self.clip_model.encode(image_obj).tolist()
            self.image_collection.add(
                embeddings=[embedding],
                metadatas=[{"source": image_path}],
                ids=[str(uuid.uuid4())]
            )
            logger.info(f"Added image {image_path} to Image collection.")
        except Exception as e:
            logger.error(f"Failed to index image {image_path}: {e}")

    def search_abstracts(self, query, n_results=5):
        """
        Search for papers based on their abstracts.
        Returns unique paper sources and their scores.
        """
        query_embedding = self.text_model.encode([query]).tolist()
        results = self.abstract_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results

    def search_within_paper(self, query, source, n_results=3):
        """
        Search for the most relevant snippets within a specific paper.
        """
        query_embedding = self.text_model.encode([query]).tolist()
        results = self.pdf_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"source": source}
        )
        return results

    def search_text(self, query, n_results=5):
        """
        Standard Search for compatibility.
        """
        query_embedding = self.text_model.encode([query]).tolist()
        results = self.pdf_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results

    def search_image(self, query_text, n_results=5):

        query_embedding = self.clip_model.encode([query_text]).tolist()
        results = self.image_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        #  # 3. 解析并格式化结果
        # formatted_results = []
        
        # # ChromaDB 返回的结果通常是嵌套列表，需要取索引 [0]
        # ids = results['ids'][0]
        # distances = results['distances'][0]
        # metadatas = results['metadatas'][0]

        # for i in range(len(ids)):
        #     # 将距离（Distance）转换为相似度分数（Score）
        #     # 这里的计算取决于你创建集合时用的距离算法（如 'cosine' 或 'l2'）
        #     # 如果是 cosine，distance 越小越相似，score = 1 - distance
        #     score = max(0, 1 - distances[i]) 
            
        #     formatted_results.append({
        #         "id": ids[i],
        #         "score": round(float(score), 4),
        #         "file_path": metadatas[i].get("file_path", ""),
        #         "metadata": metadatas[i]
        #     })

        # # 4. 按分数从高到低排序（Chroma 默认其实已经排好了，这里是双重保险）
        # formatted_results.sort(key=lambda x: x['score'], reverse=True)

        # return formatted_results

        return results
