import fitz  # PyMuPDF
from PIL import Image
import os
from utils.logger import logger

class PdfProcessor:
    @staticmethod
    # def read_pdf(file_path):
    #     """
    #     Reads a PDF file and returns a list of dictionaries, 
    #     each containing text content and page number.
    #     """
    #     doc_content = []
    #     try:
    #         doc = fitz.open(file_path)
    #         for page_num, page in enumerate(doc):
    #             text = page.get_text()
    #             if text.strip():
    #                 doc_content.append({
    #                     "text": text,
    #                     "page": page_num + 1,
    #                     "source": file_path
    #                 })
    #         logger.info(f"Successfully read {len(doc_content)} pages from {file_path}")
    #         return doc_content
    #     except Exception as e:
    #         logger.error(f"Error reading PDF {file_path}: {e}")
    #         return []
    
    def read_pdf(file_path):
        """
        Reads a PDF file and returns a list of dictionaries, 
        each containing text content and page number.
        Also attempts to extract an abstract for semantic indexing.
        """
        doc_content = []
        abstract = ""
        try:
            doc = fitz.open(file_path)
            # Try to find abstract in the first 2 pages
            for page_num in range(min(2, len(doc))):
                page_text = doc[page_num].get_text()
                if not abstract:
                    # Simple heuristic: find text between 'Abstract' and first section
                    # or just take a chunk starting with 'Abstract'
                    import re
                    # Look for 'Abstract' (case insensitive)
                    match = re.search(r'(?i)abstract[:\s\n]+([\s\S]+?)(?=\n\s*(?:1\.|Introduction|I\.|KEYWORDS))', page_text)
                    if match:
                        abstract = match.group(1).strip()
                    elif 'abstract' in page_text.lower():
                        # Fallback: just take a few hundred words after 'abstract'
                        parts = re.split(r'(?i)abstract', page_text)
                        if len(parts) > 1:
                            abstract = parts[1].strip()[:1500]
                
                print("ZWK: Extracted abstract:", abstract)

            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    doc_content.append({
                        "text": text,
                        "page": page_num + 1,
                        "source": file_path,
                        "abstract": abstract if page_num == 0 else "" # Attach abstract to first page info
                    })
            logger.info(f"Successfully read {len(doc_content)} pages from {file_path}. Abstract found: {bool(abstract)}")
            return doc_content
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return []

class ImageProcessor:
    @staticmethod
    def load_image(file_path):
        """
        Loads an image from the file path.
        """
        try:
            image = Image.open(file_path).convert("RGB")
            logger.info(f"Successfully loaded image: {file_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None
