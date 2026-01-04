import ollama
from utils.logger import logger

class TextLLM:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        logger.info(f"Initialized TextLLM with model: {model_name}")

    def classify_paper(self, text, topics):
        """
        Classifies paper content using an LLM for better reasoning.
        """
        # Truncate text to avoid token limits (Ollama handles context, but let's be safe/fast)
        context = text[:2000]
        
        prompt = f"""
        You are an expert academic assistant.
        Reflect on the following paper content:
        
        "{context}..."
        
        Classify this paper into EXACTLY ONE of the following topics: {', '.join(topics)}.
        Do not output any explanation. Only output the topic name.
        """
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            answer = response['message']['content'].strip()
            
            # Basic cleanup if model chats too much
            for topic in topics:
                if topic.lower() in answer.lower():
                    return topic
            
            return answer 
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            return "Uncategorized"
