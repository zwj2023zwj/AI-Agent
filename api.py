import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.processor import PdfProcessor, ImageProcessor
from core.indexer import VectorStore
from core.classifier import TopicClassifier
from core.vision_llm import VisionLLM
from core.text_llm import TextLLM
from utils.logger import logger

app = FastAPI(title="Local Multimodal AI Agent API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = "data"
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
STATIC_DIR = "static"

os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize Core Components
# Global instances for better performance (one-time model loading)
try:
    vector_store = VectorStore()
    topic_classifier = TopicClassifier()
    # Lazy load Vision LLM to save memory if not used
    vision_llm = None
except Exception as e:
    logger.error(f"Failed to initialize backends: {e}")
    # We'll re-try or handle errors in endpoints

# Models
class SearchQuery(BaseModel):
    query: str
    n_results: Optional[int] = 5

class QuestionQuery(BaseModel):
    question: str

# Endpoints
@app.post("/api/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    topics: Optional[str] = Form(None),
    use_llm: bool = Form(False),
    llm_model: str = Form("llama3")
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(PAPERS_DIR, f"{file_id}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Read PDF
        content = PdfProcessor.read_pdf(temp_path)
        if not content:
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail="Failed to read PDF content.")

        # 2. Classify and Move
        final_path = temp_path
        assigned_topic = "Uncategorized"
        
        if topics:
            topic_list = [t.strip() for t in topics.split(",") if t.strip()]
            if topic_list:
                full_text = " ".join([c['text'] for c in content[:3]])
                
                if use_llm:
                    classifier_llm = TextLLM(model_name=llm_model)
                    assigned_topic = classifier_llm.classify_paper(full_text, topic_list)
                else:
                    assigned_topic = topic_classifier.classify(full_text, topic_list)
                
                topic_dir = os.path.join(PAPERS_DIR, assigned_topic)
                os.makedirs(topic_dir, exist_ok=True)
                
                new_path = os.path.join(topic_dir, os.path.basename(temp_path))
                shutil.move(temp_path, new_path)
                final_path = new_path
                
                for item in content:
                    item['source'] = final_path

        # 3. Index
        vector_store.add_pdf_content(content)
        
        return {
            "status": "success",
            "filename": file.filename,
            "topic": assigned_topic,
            "path": final_path
        }
    except Exception as e:
        logger.error(f"Error in upload_paper: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers/unorganized")
async def list_unorganized_papers():
    """
    Lists PDF files directly in the papers directory that are not yet classified into subfolders.
    """
    try:
        files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(PAPERS_DIR, f))]
        
        # Also return basic info for each file
        unorganized = []
        for f in files:
            path = os.path.join(PAPERS_DIR, f)
            stats = os.stat(path)
            unorganized.append({
                "filename": f,
                "size": stats.st_size,
                "path": path
            })
        return unorganized
    except Exception as e:
        logger.error(f"Error listing unorganized papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BatchOrganizeRequest(BaseModel):
    topics: str
    use_llm: bool = False
    llm_model: str = "llama3"

@app.post("/api/papers/batch_organize")
async def batch_organize_papers(request: BatchOrganizeRequest):
    """
    Classifies and moves all PDFs directly in the papers directory.
    """
    try:
        files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(PAPERS_DIR, f))]
        
        results = []
        topic_list = [t.strip() for t in request.topics.split(",") if t.strip()]
        
        for f in files:
            path = os.path.join(PAPERS_DIR, f)
            # Reuse the classification logic from add_paper
            # We skip the "add to vector store" if it's already indexed? 
            # Or just call add_paper logic. The add_paper logic is already robust.
            
            # Since add_paper is a function in main.py but we re-implemented logic in api.py,
            # let's extract the core logic or just reuse what we have here.
            
            try:
                content = PdfProcessor.read_pdf(path)
                if not content:
                    results.append({"filename": f, "status": "failed", "detail": "Read error"})
                    continue

                full_text = " ".join([c['text'] for c in content[:3]])
                
                if request.use_llm:
                    classifier_llm = TextLLM(model_name=request.llm_model)
                    assigned_topic = classifier_llm.classify_paper(full_text, topic_list)
                else:
                    assigned_topic = topic_classifier.classify(full_text, topic_list)
                
                topic_dir = os.path.join(PAPERS_DIR, assigned_topic)
                os.makedirs(topic_dir, exist_ok=True)
                
                new_path = os.path.join(topic_dir, f)
                shutil.move(path, new_path)
                
                # Update source in content for indexing
                for item in content:
                    item['source'] = new_path
                
                # Re-index (or index for first time)
                vector_store.add_pdf_content(content)
                
                results.append({"filename": f, "status": "success", "topic": assigned_topic})
            except Exception as inner_e:
                results.append({"filename": f, "status": "failed", "detail": str(inner_e)})
        
        return results
    except Exception as e:
        logger.error(f"Error in batch_organize_papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/papers/search")
async def search_papers(query: SearchQuery):
    try:
        # 1. First Stage: Search by Abstracts to find relevant papers
        # We request a few more results than n_results to allow for some variety
        abs_results = vector_store.search_abstracts(query.query, n_results=query.n_results or 5)
        
        formatted_results = []
        
        if abs_results and 'metadatas' in abs_results and abs_results['metadatas'][0]:
            metadatas = abs_results['metadatas'][0]
            distances = abs_results['distances'][0]
            documents = abs_results['documents'][0] # This is the abstract itself
            
            for abstract_text, meta, dist in zip(documents, metadatas, distances):
                source = meta.get('source', 'Unknown')
                
                # 2. Second Stage: Find the most relevant snippet WITHIN this specific paper
                # This ensures the snippet is actually relevant to the query even if chosen by abstract
                snippet_results = vector_store.search_within_paper(query.query, source, n_results=1)
                
                content = ""
                page = "Unknown"
                
                if snippet_results and snippet_results['documents'][0]:
                    content = snippet_results['documents'][0][0]
                    page = snippet_results['metadatas'][0][0].get('page', 'Unknown')
                else:
                    # Fallback to abstract if no snippet found in content collection
                    content = abstract_text[:500] + "..."
                
                formatted_results.append({
                    "content": content,
                    "abstract": abstract_text[:300] + "...",
                    "source": os.path.basename(source),
                    "full_path": source,
                    "page": page,
                    "score": 1 - dist  # Score from the abstract match
                })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error in search_papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)):
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in supported_exts:
        raise HTTPException(status_code=400, detail=f"Supported formats: {supported_exts}")
    
    file_id = str(uuid.uuid4())
    save_path = os.path.join(IMAGES_DIR, f"{file_id}{ext}")
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        img = ImageProcessor.load_image(save_path)
        if img:
            vector_store.add_image(save_path, img)
            return {"status": "success", "filename": file.filename, "path": save_path}
        else:
            os.remove(save_path)
            raise HTTPException(status_code=500, detail="Failed to process image.")
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/images/search")
async def search_images(query: SearchQuery):
    try:
        results = vector_store.search_image(query.query, n_results=2)
        formatted_results = []
        
        if results and 'metadatas' in results and results['metadatas'][0]:
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for meta, dist in zip(metadatas, distances):
                path = meta.get('source', 'Unknown')
                
                # Normalize path for frontend (serve via /data)
                # Ensure it's relative to project root
                try:
                    rel_path = os.path.relpath(path, os.getcwd())
                    # Ensure it uses forward slashes
                    web_path = rel_path.replace("\\", "/")
                except:
                    web_path = path.replace("\\", "/")

                formatted_results.append({
                    "source": os.path.basename(path),
                    "full_path": web_path,
                    "score": 1 - dist
                })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error in search_images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision/describe")
async def describe_image(path: str = Form(...)):
    global vision_llm
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found at path.")
    
    try:
        if vision_llm is None:
            vision_llm = VisionLLM()
        description = vision_llm.describe_image(path)
        return {"description": description}
    except Exception as e:
        logger.error(f"Error in describe_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision/ask")
async def ask_image(path: str = Form(...), question: str = Form(...)):
    global vision_llm
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found at path.")
    
    try:
        if vision_llm is None:
            vision_llm = VisionLLM()
        answer = vision_llm.visual_qa(path, question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in ask_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    try:
        return {
            "papers_indexed": vector_store.pdf_collection.count(),
            "images_indexed": vector_store.image_collection.count(),
        }
    except:
        return {"papers_indexed": 0, "images_indexed": 0}

# Serve data directory for image previews
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
# Serve static files
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Start the Local Multimodal AI Agent API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
