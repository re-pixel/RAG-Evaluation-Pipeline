from sentence_transformers import SentenceTransformer, util

class EmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
        self.name = "multi-qa-mpnet-base-dot-v1"
        
    def __call__(self, input):
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()