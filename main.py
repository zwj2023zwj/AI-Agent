import argparse
import os
import shutil
import sys
from core.processor import PdfProcessor, ImageProcessor
from core.indexer import VectorStore
from core.classifier import TopicClassifier
from core.vision_llm import VisionLLM
from core.text_llm import TextLLM
from utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: add_paper
    add_paper_parser = subparsers.add_parser("add_paper", help="Add and classify a paper")
    add_paper_parser.add_argument("path", help="Path to the PDF file")
    add_paper_parser.add_argument("--topics", help="Comma-separated topics (e.g. 'CV,NLP,RL')")
    add_paper_parser.add_argument("--use-llm", action="store_true", help="Use LLM for classification (requires Ollama)")
    add_paper_parser.add_argument("--llm-model", default="llama3", help="Ollama model name (default: llama3)")

    # Command: search_paper
    search_paper_parser = subparsers.add_parser("search_paper", help="Search for papers")
    search_paper_parser.add_argument("query", help="Search query")
    search_paper_parser.add_argument("--list-only", action="store_true", help="Return list of relevant files only")

    # Command: search_image
    search_image_parser = subparsers.add_parser("search_image", help="Search for images")
    search_image_parser.add_argument("query", help="Search query")

    # Command: scan_images (Extra utility to build image db)
    scan_img_parser = subparsers.add_parser("scan_images", help="Index all images in a folder")
    scan_img_parser.add_argument("folder", help="Folder containing images")
    
    # Command: organize_folder
    org_folder_parser = subparsers.add_parser("organize_folder", help="Organize PDFs in a folder")
    org_folder_parser.add_argument("folder", help="Folder to organize")
    org_folder_parser.add_argument("--topics", help="Comma-separated topics (e.g. 'CV,NLP,RL')", required=True)

    # Command: describe_image
    desc_img_parser = subparsers.add_parser("describe_image", help="Generate caption for an image (Florence-2)")
    desc_img_parser.add_argument("path", help="Path to the image")

    # Command: ask_image
    ask_img_parser = subparsers.add_parser("ask_image", help="Ask a question about an image (Florence-2)")
    ask_img_parser.add_argument("path", help="Path to the image")
    ask_img_parser.add_argument("question", help="Question to ask")

    args = parser.parse_args()

    if args.command == "add_paper":
        add_paper(args.path, args.topics, args.use_llm, args.llm_model)
    elif args.command == "search_paper":
        search_paper(args.query, args.list_only)
    elif args.command == "search_image":
        search_image(args.query)
    elif args.command == "scan_images":
        scan_images(args.folder)
    elif args.command == "organize_folder":
        organize_folder(args.folder, args.topics)
    elif args.command == "describe_image":
        vision_llm = VisionLLM()
        answers = vision_llm.describe_image(args.path)
        logger.info(answers) 
    elif args.command == "ask_image":
        vision_llm = VisionLLM()
        answers = vision_llm.visual_qa(args.path, args.question)
        logger.info(answers)
    else:
        parser.print_help()

def add_paper(path, topics_str, use_llm=False, llm_model="llama3"):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return

    # 1. Read PDF
    logger.info(f"Processing {path}...")
    content = PdfProcessor.read_pdf(path)
    if not content:
        logger.error("Failed to read PDF content.")
        return

    # 2. Classify and Move (if topics provided)
    final_path = path
    if topics_str:
        topics = [t.strip() for t in topics_str.split(",") if t.strip()]
        if topics:
            # Combine first few pages for classification context
            full_text = " ".join([c['text'] for c in content[:3]])
            
            if use_llm:
                logger.info(f"Classifying using LLM ({llm_model})...")
                classifier_llm = TextLLM(model_name=llm_model)
                topic = classifier_llm.classify_paper(full_text, topics)
            else:
                logger.info("Classifying using Embeddings (fast)...")
                classifier = TopicClassifier()
                topic = classifier.classify(full_text, topics)
            
            # Create topic directory
            base_dir = os.path.dirname(path)
            topic_dir = os.path.join(base_dir, topic)
            
            if not os.path.exists(topic_dir):
                os.makedirs(topic_dir, exist_ok=True)
            
            # Move file
            filename = os.path.basename(path)
            new_path = os.path.join(topic_dir, filename)
            try:
                shutil.move(path, new_path)
                logger.info(f"Moved to {new_path}")
                final_path = new_path
                
                # Update source path in content
                for item in content:
                    item['source'] = final_path
            except Exception as e:
                logger.error(f"Failed to move file: {e}")

    # 3. Index Content
    indexer = VectorStore()
    indexer.add_pdf_content(content)
    logger.info("Paper added and indexed successfully.")

def search_paper(query, list_only=False):
    indexer = VectorStore()
    results = indexer.search_text(query, n_results=10 if list_only else 5)

    # Format results
    # results is a dict with keys: ids, embeddings, documents, metadatas, distances
    if results and 'documents' in results and results['documents'][0]:
        if list_only:
            print(f"\nRelevant Files for '{query}':\n")
            metadatas = results['metadatas'][0]
            # Extract unique sources
            unique_files = set()
            for meta in metadatas:
                source = meta.get('source', 'Unknown')
                unique_files.add(source)
            
            for i, file in enumerate(unique_files):
                print(f"{i+1}. {file}")
            print("")
        else:
            print(f"\nSearch Results for '{query}':\n")
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                print(f"--- Result {i+1} ---")
                print(f"File: {meta.get('source', 'Unknown')}")
                print(f"Page: {meta.get('page', 'Unknown')}")
                print(f"Snippet: {doc[:200].replace(chr(10), ' ')}...\n")
    else:
        print("No results found.")

def search_image(query):
    indexer = VectorStore()
    results = indexer.search_image(query, n_results=5)
    
    print("HHH")
    if results and 'metadatas' in results and results['metadatas'][0]:
        print(f"\nImage Search Results for '{query}':\n")
        metadatas = results['metadatas'][0]
        
        for i, meta in enumerate(metadatas):
            print(f"--- Result {i+1} ---")
            print(f"File: {meta.get('source', 'Unknown')}\n")
    else:
        print("No images found.")

def scan_images(folder):

    if not os.path.exists(folder):
        logger.error(f"Folder not found: {folder}")
        return
        
    indexer = VectorStore()
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_exts:
                path = os.path.join(root, file)
                img = ImageProcessor.load_image(path)
                if img:
                    indexer.add_image(path, img)
                    count += 1
    logger.info(f"Indexed {count} images from {folder}")

def organize_folder(folder, topics_str):
    if not os.path.exists(folder):
        logger.error(f"Folder not found: {folder}")
        return

    # Iterate direct children PDF files
    files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(files)} PDFs in {folder}")
    
    for f in files:
        path = os.path.join(folder, f)
        add_paper(path, topics_str)

if __name__ == "__main__":
    main()
