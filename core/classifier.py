from sentence_transformers import SentenceTransformer, util
import torch
from utils.logger import logger

class TopicClassifier:
    def __init__(self):
        # We reuse the same model as the Indexer for consistency and efficiency
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def classify(self, text, topics):
        """
        Classifies the text into one of the provided topics.
        text: str (content of the paper)
        topics: list of str (e.g. ["CV", "NLP"])
        """
        if not text or not topics:
            return "Uncategorized"
        
        # Embed the text (truncate if too long, though MiniLM handles truncation)
        # We take the first 512 words roughly to get the gist
        short_text = " ".join(text.split()[:512])
        
        text_embedding = self.model.encode(short_text, convert_to_tensor=True)
        topic_embeddings = self.model.encode(topics, convert_to_tensor=True)

        cosine_scores = util.cos_sim(text_embedding, topic_embeddings)
        
        # Find the index of the best matching topic
        best_topic_idx = torch.argmax(cosine_scores).item()
        best_topic = topics[best_topic_idx]
        
        logger.info(f"Classified document as: {best_topic}")
        return best_topic
