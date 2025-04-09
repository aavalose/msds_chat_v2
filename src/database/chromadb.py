import streamlit as st
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InMemoryVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.vectors = None

    def add(self, ids, documents, metadatas):
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.vectors = self.vectorizer.fit_transform(self.documents)

    def count(self):
        """Return the number of documents in the store"""
        return len(self.documents)

    def query(self, query_texts, n_results=1):
        if self.vectors is None:
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        # Handle single query text or list of query texts
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        
        results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        for query_text in query_texts:
            query_vector = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # Get top n_results indices
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            # Format results to match ChromaDB's return format
            results['ids'].append([self.ids[i] for i in top_indices])
            results['documents'].append([self.documents[i] for i in top_indices])
            results['metadatas'].append([self.metadatas[i] for i in top_indices])
            results['distances'].append([1 - similarities[i] for i in top_indices])  # Convert similarity to distance
        
        return results

@st.cache_resource
def init_chroma():
    try:
        logger.info("Initializing in-memory vector store")
        vector_store = InMemoryVectorStore()
        return vector_store, None  # Return None for embedding_function to maintain compatibility
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        st.error(f"Error initializing vector store: {str(e)}")
        raise

@st.cache_resource
def load_and_index_json_data(_vector_store, _embedding_function, collection_name="msds_program_qa"):
    try:
        logger.info(f"Loading and indexing data for collection: {collection_name}")
        
        # Load data from context.json file
        try:
            context_path = os.path.join(os.getcwd(), "data", "context.json")
            logger.info(f"Loading context data from: {context_path}")
            
            with open(context_path, "r") as f:
                context_data = json.load(f)
            
            # Generate documents for vector store from JSON data
            documents = []
            metadatas = []
            ids = []
            counter = 0
            
            # Process each category in context.json
            for category, data in context_data.items():
                # If the category has QA pairs, add them to the collection
                if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
                    for qa_pair in data["qa_pairs"]:
                        if "question" in qa_pair and "answer" in qa_pair:
                            documents.append(qa_pair["question"])
                            metadatas.append({
                                "category": category,
                                "answer": qa_pair["answer"],
                                "type": "qa_pair"
                            })
                            ids.append(f"{category.lower().replace(' ', '_')}_{counter}")
                            counter += 1
                
                # Also create broader category-based questions
                category_questions = [
                    f"Tell me about {category}",
                    f"What is the {category} like?",
                    f"Information about {category}"
                ]
                
                # Create a summary of the category data
                summary = json.dumps(data, ensure_ascii=False)
                if len(summary) > 1000:  # If too long, create a shorter version
                    # Remove qa_pairs for the summary to keep it focused on structured data
                    summary_data = {k: v for k, v in data.items() if k != 'qa_pairs'}
                    summary = json.dumps(summary_data, ensure_ascii=False)
                
                for question in category_questions:
                    documents.append(question)
                    metadatas.append({
                        "category": category,
                        "answer": summary,
                        "type": "category_summary"
                    })
                    ids.append(f"{category.lower().replace(' ', '_')}_summary_{counter}")
                    counter += 1
            
            # Now add all the data to the vector store
            if documents:
                try:
                    _vector_store.add(ids, documents, metadatas)
                    logger.info(f"Successfully added {len(documents)} documents to vector store")
                except Exception as e:
                    logger.error(f"Failed to add documents to vector store: {str(e)}")
                    raise
            else:
                logger.warning("No documents were created from JSON data")
                st.warning("No documents were created from JSON data")

        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            st.error(f"Error loading JSON data: {str(e)}")
            raise

        return _vector_store
    except Exception as e:
        logger.error(f"Error in load_and_index_json_data: {str(e)}")
        st.error(f"Error initializing QA collection from JSON: {str(e)}")
        raise
