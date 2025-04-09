import streamlit as st
import json
from src.utils.preprocessing import preprocess_query
from src.database.chromadb import init_chroma, load_and_index_json_data

# Initialize ChromaDB and collection
chroma_client, embedding_function = init_chroma()
qa_collection = load_and_index_json_data(chroma_client, embedding_function)

def find_most_similar_question(user_input, similarity_threshold=0.3):
    try:
        if qa_collection.count() == 0:
            return [], [], 0.0
        
        # Process the query to get categories
        processed_input, primary_category, all_categories = preprocess_query(user_input)
        
        # Query ChromaDB without filter condition
        results = qa_collection.query(
            query_texts=[processed_input],
            n_results=5  # Get top 5 results
        )
        
        if not results['documents'][0]:
            return [], [], 0.0
        
        # Collect all questions and answers that meet the threshold
        matching_questions = []
        matching_answers = []
        best_similarity = 0.0
        
        for i, distance in enumerate(results['distances'][0]):
            similarity = 1 - distance  # Convert distance back to similarity
            if similarity >= similarity_threshold:
                matching_questions.append(results['documents'][0][i])
                
                # Get the answer and handle JSON if needed
                metadata = results['metadatas'][0][i]
                answer = metadata.get('answer', '')
                answer_type = metadata.get('type', '')
                
                if answer_type == 'category_summary':
                    # For category summaries that might be JSON strings, format them nicely
                    try:
                        answer_data = json.loads(answer)
                        # Format based on the structure of the data
                        if isinstance(answer_data, dict):
                            formatted_answer = f"Here's information about {metadata.get('category', 'this topic')}:\n\n"
                            # Exclude qa_pairs from the formatted output to avoid duplication
                            for k, v in answer_data.items():
                                if k != 'qa_pairs':
                                    if isinstance(v, dict):
                                        formatted_answer += f"{k}:\n"
                                        for sub_k, sub_v in v.items():
                                            formatted_answer += f"  - {sub_k}: {sub_v}\n"
                                    elif isinstance(v, list):
                                        formatted_answer += f"{k}:\n"
                                        for item in v:
                                            formatted_answer += f"  - {item}\n"
                                    else:
                                        formatted_answer += f"{k}: {v}\n"
                            matching_answers.append(formatted_answer)
                        else:
                            matching_answers.append(str(answer_data))
                    except json.JSONDecodeError:
                        # If it's not valid JSON, use as is
                        matching_answers.append(answer)
                else:
                    # For direct QA pairs, use the answer as is
                    matching_answers.append(answer)
                
                best_similarity = max(best_similarity, similarity)
        
        return matching_questions, matching_answers, best_similarity
            
    except Exception as e:
        st.error(f"Error in find_most_similar_question: {str(e)}")
        return [], [], 0.0
