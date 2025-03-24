from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class SimilarityClient:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.questions = []
        self.answers = []
        self.vectors = None
        self._load_qa_data()

    def _load_qa_data(self):
        try:
            qa_df = pd.read_csv("Questions_and_Answers.csv")
            self.questions = qa_df['Question'].tolist()
            self.answers = qa_df['Answer'].tolist()
            
            # Create TF-IDF vectors for questions
            if self.questions:
                self.vectors = self.vectorizer.fit_transform(self.questions)
            
        except Exception as e:
            print(f"Error loading QA data: {str(e)}")
            # Create a minimal dataset if loading fails
            self.questions = ["Default question"]
            self.answers = ["Please contact the MSDS program office for more information."]
            self.vectors = self.vectorizer.fit_transform(self.questions)

    def find_similar_question(self, query, similarity_threshold=0.3):
        if not self.questions:
            return None, None, 0.0

        # Transform the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Find the best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= similarity_threshold:
            return (
                self.questions[best_idx],
                self.answers[best_idx],
                float(best_similarity)
            )
        
        return None, None, 0.0 