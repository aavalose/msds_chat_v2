import chromadb
from chromadb.utils import embedding_functions
import os
import pandas as pd

class ChromaDBClient:
    def __init__(self, db_path="chroma_db"):
        self.db_path = db_path
        self.client = self._init_client()
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self._init_collection()

    def _init_client(self):
        os.makedirs(self.db_path, exist_ok=True)
        return chromadb.PersistentClient(path=self.db_path)

    def _init_collection(self):
        try:
            collection = self.client.get_collection(
                name="msds_program_qa",
                embedding_function=self.embedding_function
            )
        except:
            collection = self.client.create_collection(
                name="msds_program_qa",
                embedding_function=self.embedding_function
            )
            self._load_qa_data(collection)
        return collection

    def _load_qa_data(self, collection):
        qa_df = pd.read_csv("Questions_and_Answers.csv")
        collection.add(
            ids=[str(i) for i in qa_df.index.tolist()],
            documents=qa_df['Question'].tolist(),
            metadatas=qa_df[['Answer']].to_dict(orient='records')
        )

    def find_similar_question(self, query, similarity_threshold=0.3):
        if self.collection.count() == 0:
            return None, None, 0.0

        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )

        if not results['documents'][0]:
            return None, None, 0.0

        best_similarity = 0.0
        best_question = None
        best_answer = None

        for i, distance in enumerate(results['distances'][0]):
            similarity = 1 - distance
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_question = results['documents'][0][i]
                best_answer = results['metadatas'][0][i]['Answer']

        return best_question, best_answer, best_similarity 