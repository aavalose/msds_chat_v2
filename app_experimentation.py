import sys
import os

try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import numpy as np
from datetime import datetime
import pandas as pd
import json
import google.generativeai as genai
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions


# Handle missing API key safely
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please configure it in your Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Add MongoDB configuration near the top of the file
# Use st.secrets to store your MongoDB connection string
MONGO_CONNECTION_STRING = st.secrets.get("MONGO_CONNECTION_STRING")
if not MONGO_CONNECTION_STRING:
    st.error("MongoDB connection string not found. Please configure it in your Streamlit secrets.")
    st.stop()

# Initialize ChromaDB client
@st.cache_resource
def init_chroma():
    try:
        # Create a persistent directory for ChromaDB
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize the client with persistence
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Use ChromaDB's default embedding function
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        return chroma_client, embedding_function
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        raise e

# Add this function to handle collection creation and data loading
# Update the function signature with leading underscores
@st.cache_resource
def load_and_index_json_data(_chroma_client, _embedding_function, collection_name="msds_program_qa"):
    try:
        # Delete existing collection if it exists
        try:
            _chroma_client.delete_collection(name=collection_name)
            if st.session_state.get('debug_mode', False):
                st.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            # Collection might not exist, which is fine
            pass
            
        # Create new collection
        qa_collection = _chroma_client.create_collection(
            name=collection_name,
            embedding_function=_embedding_function
        )

        # Load data from context.json file
        try:
            with open("context.json", "r") as f:
                context_data = json.load(f)
            
            # Generate documents for ChromaDB from JSON data
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
                                "Category": category,
                                "Answer": qa_pair["answer"],
                                "Type": "qa_pair"
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
                        "Category": category,
                        "Answer": summary,
                        "Type": "category_summary"
                    })
                    ids.append(f"{category.lower().replace(' ', '_')}_summary_{counter}")
                    counter += 1
            
            # Now add all the data to ChromaDB
            if documents:
                qa_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                if st.session_state.get('debug_mode', False):
                    st.success(f"Successfully indexed {len(documents)} questions from JSON data")
            else:
                st.warning("No documents were created from JSON data")

        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
            raise e

        return qa_collection
    except Exception as e:
        st.error(f"Error initializing QA collection from JSON: {str(e)}")
        raise e

# Initialize ChromaDB and collection
try:
    chroma_client, embedding_function = init_chroma()
    qa_collection = load_and_index_json_data(chroma_client, embedding_function)
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {str(e)}")
    st.stop()

# Configure Gemini model
@st.cache_resource
def load_gemini_model():
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

gemini_model = load_gemini_model()

# Initialize MongoDB client
@st.cache_resource
def init_mongodb():
    try:
        # Add SSL and connection pool configurations
        client = MongoClient(
            MONGO_CONNECTION_STRING,
            tls=True,
            tlsAllowInvalidCertificates=False,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True,
            maxPoolSize=50
        )
        
        # Test the connection
        client.admin.command('ping')
        
        db = client.MSDSchatbot
        return db.conversations
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

conversations_collection = init_mongodb()

# 1. Add these imports at the top of your file
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. Add this new function for calculating cosine similarity
def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two text strings using TF-IDF"""
    if not text1 or not text2:
        return 0.0
        
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

# 3. Update the save_conversation function to include response similarity
def save_conversation(session_id, user_message, bot_response, response_time, metrics=None, response_similarity=None):
    if conversations_collection is None:
        st.error("MongoDB connection not available")
        return
        
    try:
        conversation = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "user_message": user_message,
            "bot_response": bot_response,
            "feedback": None,
            "similarity_score": st.session_state.debug_similarity,
            "matched_question": st.session_state.debug_matched_question,
            "response_time_seconds": response_time,
            "response_similarity": response_similarity  # Add this field
        }
        
        # Add metrics if available
        if metrics:
            conversation["metrics"] = metrics
            
        result = conversations_collection.insert_one(conversation)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error saving conversation to MongoDB: {str(e)}")
        return None

# Add this function after save_conversation
def update_feedback(conversation_id, feedback):
    if conversations_collection is None:
        st.error("MongoDB connection not available")
        return
        
    try:
        conversations_collection.update_one(
            {"_id": conversation_id},
            {"$set": {"feedback": feedback}}
        )
    except Exception as e:
        st.error(f"Error updating feedback: {str(e)}")

# Update the find_most_similar_question function
def find_most_similar_question(user_input, similarity_threshold=0.3):
    try:
        if qa_collection.count() == 0:
            return [], [], 0.0
        
        # Process the query to get categories
        processed_input, primary_category, all_categories = preprocess_query(user_input)
        
        # Query with filter for matching category if not "Other"
        filter_condition = {"Category": {"$in": all_categories}} if "Other" not in all_categories else None
        
        # Query ChromaDB
        results = qa_collection.query(
            query_texts=[processed_input],
            n_results=5,  # Get top 5 results
            where=filter_condition
        )
        
        if not results['documents'][0]:
            return [], [], 0.0
        
        # Collect all questions and answers that meet the threshold
        matching_questions = []
        matching_answers = []
        best_similarity = 0.0
        
        for i, distance in enumerate(results['distances'][0]):
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                matching_questions.append(results['documents'][0][i])
                
                # Get the answer and handle JSON if needed
                answer = results['metadatas'][0][i]['Answer']
                answer_type = results['metadatas'][0][i]['Type']
                
                if answer_type == 'category_summary':
                    # For category summaries that might be JSON strings, format them nicely
                    try:
                        answer_data = json.loads(answer)
                        # Format based on the structure of the data
                        if isinstance(answer_data, dict):
                            formatted_answer = "Here's information about " + results['metadatas'][0][i]['Category'] + ":\n\n"
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
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write("Top matches:")
            st.write(f"Categories: {all_categories}")
            st.write(f"Number of matching questions: {len(matching_questions)}")
            for i, (q, a, dist) in enumerate(zip(matching_questions, matching_answers, results['distances'][0])):
                sim = 1 - dist
                st.write(f"{i+1}. Question: {q}")
                st.write(f"   Similarity: {sim:.3f}")
                st.write(f"   Answer: {a[:100]}...")  # Show first 100 chars of answer
        
        return matching_questions, matching_answers, best_similarity
            
    except Exception as e:
        st.error(f"Error in find_most_similar_question: {str(e)}")
        return [], [], 0.0

# Enhance the preprocess_query function
def preprocess_query(query):
    processed_query = query.lower().strip()
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Analyze this question and return up to THREE most relevant categories from the following list, ordered by relevance:
        - Application Process: Questions about how to apply, deadlines, interviews, and application components
        - Admission Requirements: Questions about prerequisites, qualifications, and requirements
        - Financial Aid & Scholarships: Questions about funding, scholarships, and financial assistance
        - International Students: Questions specific to international student needs
        - Enrollment Process: Questions about post-acceptance procedures
        - Program Structure: Questions about program duration, format, and class sizes
        - Program Overview: Questions about general program information and features
        - Tuition & Costs: Questions about program costs, fees, and expenses
        - Program Preparation: Questions about preparing for the program
        - Faculty & Research: Questions about professors and research opportunities
        - Student Employment: Questions about work opportunities during the program
        - Student Services: Questions about health insurance and student support
        - Curriculum: Questions about courses and academic content
        - Practicum Experience: Questions about industry projects and partnerships
        - Career Outcomes: Questions about job placement, salaries, and career paths
        - Admission Statistics: Questions about typical GPAs, backgrounds, and work experience
        - Other: Questions that don't clearly fit into any of the above categories
        
        Examples:
        Question: "What GRE score do I need as an international student?" -> ["Application Process", "International Students"]
        Question: "How much is tuition and what scholarships are available?" -> ["Tuition & Costs", "Financial Aid & Scholarships"]
        Question: "Can I work while taking classes in the program?" -> ["Student Employment", "Program Structure"]
        Question: "Where is the nearest coffee shop?" -> ["Other"]
        
        Your question: "{query}"
        
        Return only the category names in a comma-separated list, nothing else."""
        
        response = model.generate_content(prompt)
        categories = [cat.strip() for cat in response.text.split(',')]
        primary_category = categories[0] if categories else "Other"
        
        if st.session_state.get('debug_mode', False):
            st.write(f"Detected categories: {categories}")
            
        return processed_query, primary_category, categories
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error categorizing query: {str(e)}")
        return processed_query, "Other", ["Other"]

def get_conversation_history(max_messages=5):
    """Get the recent conversation history formatted for the prompt"""
    if 'chat_history' not in st.session_state:
        return ""
    
    # Get last 5 message pairs (10 messages total)
    recent_messages = st.session_state.chat_history[-max_messages*2:]
    
    if not recent_messages:
        return ""
    
    # Format conversation history
    history = "\nRecent conversation history:\n"
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    return history

# Update the get_gemini_response function
def get_gemini_response(user_input, retrieved_questions=None, retrieved_answers=None):
    try:
        # Get conversation history
        conversation_history = get_conversation_history()
        
        # Process the query to get categories
        processed_query, primary_category, all_categories = preprocess_query(user_input)
        
        # Load general information
        general_info = open('general_info.txt', 'r').read()
        
        # Load relevant category information from context.json
        try:
            with open('context.json', 'r') as f:
                context_data = json.load(f)
            
            # Get category-specific information but exclude qa_pairs to avoid redundancy
            category_info = {}
            for category in all_categories:
                if category in context_data:
                    # Create a copy without qa_pairs
                    category_info[category] = {k: v for k, v in context_data[category].items() if k != 'qa_pairs'}
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.error(f"Error loading context data: {str(e)}")
            category_info = {}
        
        # Format QA pairs
        relevant_qa_pairs = ""
        if retrieved_questions and retrieved_answers and st.session_state.debug_similarity >= 0.3:
            if not isinstance(retrieved_questions, list):
                retrieved_questions = [retrieved_questions]
                retrieved_answers = [retrieved_answers]
            
            relevant_qa_pairs = "\n\nRelevant information from our database:\n"
            for q, a in zip(retrieved_questions, retrieved_answers):
                relevant_qa_pairs += f"Q: {q}\nA: {a}\n"
        
        # Enhanced prompt with conversation history and category information
        prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
        
        Conversation History: {conversation_history}
        
        Current user question: "{user_input}"
        Primary Category: {primary_category}
        Related Categories: {', '.join(all_categories[1:]) if len(all_categories) > 1 else 'None'}
        
        Category-specific information:
        ```
        {json.dumps(category_info, indent=2)}
        ```
        
        {relevant_qa_pairs}
        
        Instructions:
        1. Consider the conversation history when formulating your response
        2. If the user refers to previous messages, use that context
        3. Use the provided information to formulate a comprehensive response
        4. If the information contains specific facts, numbers, or requirements, preserve them exactly
        5. Focus on answering the user's specific question
        6. Use a conversational tone while maintaining accuracy
        7. If any information is missing or unclear, acknowledge it
        
        Additional context:
        {general_info}
        
        Please provide your response:"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."

# 4. Update get_bot_response to calculate response similarity
def get_bot_response(user_input):
    if not user_input.strip():
        return "Please enter a question.", None
    
    # Update to use the new return values
    processed_query, primary_category, all_categories = preprocess_query(user_input)
    
    # Get all matching questions and answers
    matched_questions, matched_answers, similarity = find_most_similar_question(user_input)
    
    # Debug information
    st.session_state.debug_similarity = similarity
    
    # Show all matched questions
    if matched_questions:
        st.session_state.debug_matched_question = "\n".join([f"{i+1}. {q}" for i, q in enumerate(matched_questions)])
    else:
        st.session_state.debug_matched_question = "No match found"
    
    # Show all matched answers
    if matched_answers:
        st.session_state.debug_matched_answer = "\n".join([f"{i+1}. {a}" for i, a in enumerate(matched_answers)])
    else:
        st.session_state.debug_matched_answer = "No answer found"
    
    st.session_state.debug_category = f"{primary_category} (Related: {', '.join(all_categories[1:])})" if len(all_categories) > 1 else primary_category
    
    # Generate response using Gemini, passing all matched Q&As
    bot_response = get_gemini_response(user_input, matched_questions, matched_answers)
    
    # Calculate response similarity if we have matched answers
    if matched_answers and len(matched_answers) > 0:
        # Use the first (most relevant) answer for similarity calculation
        response_similarity = calculate_cosine_similarity(bot_response, matched_answers[0])
        st.session_state.debug_response_similarity = response_similarity
        
        # Add to similarity history
        if 'similarity_history' not in st.session_state:
            st.session_state.similarity_history = []
            
        st.session_state.similarity_history.append({
            'query': user_input[:50] + "..." if len(user_input) > 50 else user_input,
            'retrieval_similarity': similarity,
            'response_similarity': response_similarity,
            'timestamp': datetime.now()
        })
    else:
        st.session_state.debug_response_similarity = None
    
    return bot_response


def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state variables
    for key in ['debug_matched_question', 'debug_matched_answer', 'debug_similarity', 
                'chat_history', 'session_id', 'conversation_ids', 'debug_category',
                'debug_response_similarity', 'similarity_history']:
        if key not in st.session_state:
            st.session_state[key] = "" if key not in ['chat_history', 'conversation_ids', 'similarity_history'] else []
            if key in ['debug_similarity', 'debug_response_similarity']:
                st.session_state[key] = 0.0
            elif key == 'session_id':
                st.session_state[key] = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize ChromaDB and collection with the JSON-based approach only
    try:
        chroma_client, embedding_function = init_chroma()
        qa_collection = load_and_index_json_data(chroma_client, embedding_function)
        if st.session_state.get('debug_mode', False):
            st.success("Using JSON-based approach")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {str(e)}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "About", "Debug", "Similarity Analysis"])

    with tab1:
        
        with st.sidebar:
            st.subheader("Session Management")
            st.write(f"Current Session ID: {st.session_state.session_id}")

            if st.button("Start New Session"):
                st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                st.session_state.chat_history = []
                st.rerun()
            
            st.subheader("Example Questions:")
            example_questions = [
                "What are the admission requirements for the MSDS program?",
                "How long does the MSDS program take to complete?",
                "What programming languages are taught in the program?",
                "Who are the faculty members in the MSDS program?",
                "What kind of projects do MSDS students work on?",
                "What is the tuition for the MSDS program?"
            ]
            
            for q in example_questions:
                if st.button(q, key=f"btn_{q[:20]}"): # Added unique keys for buttons
                    matched_question, matched_answer, similarity = find_most_similar_question(q)
                    bot_response = get_bot_response(q)
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                    save_conversation(st.session_state.session_id, q, bot_response, 0.0)
        
        st.subheader("Ask me about USF's MSDS program")
        user_message = st.text_input("Type your question here:", key="user_input")
        
        if st.button("Send", key="send_button") and user_message:
            with st.spinner("Thinking..."):
                start_time = datetime.now()
                bot_response = get_bot_response(user_message)
                response_time = (datetime.now() - start_time).total_seconds()
                
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                
                conversation_id = save_conversation(
                    st.session_state.session_id, 
                    user_message, 
                    bot_response,
                    response_time,
                    response_similarity=st.session_state.debug_response_similarity  # Pass similarity score for storage
                )
                if conversation_id:
                    if 'conversation_ids' not in st.session_state:
                        st.session_state.conversation_ids = []
                    st.session_state.conversation_ids.append(conversation_id)
        
        # Get chat history pairs in reverse order (newest first)
        chat_pairs = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_msg = st.session_state.chat_history[i]
                bot_msg = st.session_state.chat_history[i + 1]
                chat_pairs.append((user_msg, bot_msg))

        # Display newest messages first
        for i, (user_msg, bot_msg) in enumerate(reversed(chat_pairs)):
            # User message - right aligned
            user_container = st.container()
            with user_container:
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #007AFF;
                            color: white;
                            padding: 10px 15px;
                            border-radius: 20px;
                            margin: 5px 0;
                            max-width: 90%;
                            float: right;
                        ">
                            {user_msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.write("🧑")

            # Bot message - left aligned
            bot_container = st.container()
            with bot_container:
                col1, col2 = st.columns([1, 6])
                with col1:
                    st.write("🤖")
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #E9ECEF;
                            color: black;
                            padding: 10px 15px;
                            border-radius: 20px;
                            margin: 5px 0;
                            max-width: 90%;
                        ">
                            {bot_msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍", key=f"thumbs_up_{i}"):
                    if i < len(st.session_state.conversation_ids):
                        update_feedback(st.session_state.conversation_ids[-(i+1)], "positive")
                        st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎", key=f"thumbs_down_{i}"):
                    if i < len(st.session_state.conversation_ids):
                        update_feedback(st.session_state.conversation_ids[-(i+1)], "negative")
                        st.success("Thank you for your feedback!")

    with tab3:
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if st.session_state.debug_mode:
            st.write("Last Query Debug Info:")
            st.write(f"Category: {st.session_state.debug_category}")
            st.write(f"Similarity Score: {st.session_state.debug_similarity:.3f}")
            st.write(f"Matched Question: {st.session_state.debug_matched_question}")
            st.write(f"Matched Answer: {st.session_state.debug_matched_answer}")
     # Add the new tab4 content
    with tab4:
        st.header("Response Similarity Analysis")
        st.write("""
        This tab analyzes how closely the chatbot's generated responses match 
        the reference answers it was provided during the retrieval-augmented generation (RAG) process.
        """)
        
        # Display current similarity metrics if available
        if hasattr(st.session_state, 'debug_response_similarity') and st.session_state.debug_response_similarity is not None:
            st.metric(
                label="Current Response Similarity", 
                value=f"{st.session_state.debug_response_similarity:.3f}"
            )
            
            # Visualization of similarity threshold
            st.subheader("Similarity Interpretation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Low Similarity (< 0.3)")
                st.progress(min(max(st.session_state.debug_response_similarity, 0) / 0.3, 1.0))
            with col2:
                st.write("Medium Similarity (0.3-0.7)")
                similarity_val = st.session_state.debug_response_similarity
                if similarity_val < 0.3:
                    st.progress(0.0)
                elif similarity_val > 0.7:
                    st.progress(1.0)
                else:
                    st.progress((similarity_val - 0.3) / 0.4)
            with col3:
                st.write("High Similarity (> 0.7)")
                st.progress(max(min((st.session_state.debug_response_similarity - 0.7) / 0.3, 1.0), 0.0))
                
        # Show comparison between latest response and matched answer
        st.subheader("Latest Response Comparison")
        
        if (st.session_state.chat_history and len(st.session_state.chat_history) >= 2 and 
            st.session_state.debug_matched_answer):
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Generated Response:")
                st.text_area(
                    "Chatbot's response", 
                    st.session_state.chat_history[-1]["content"] if st.session_state.chat_history[-1]["role"] == "assistant" else "",
                    height=200
                )
            with col2:
                st.write("Most Similar Reference Answer:")
                # Extract first matched answer
                first_answer = st.session_state.debug_matched_answer.split("\n")[0]
                if first_answer.startswith("1. "):
                    first_answer = first_answer[3:]  # Remove the "1. " prefix
                st.text_area("Reference answer", first_answer, height=200)
        else:
            st.info("Send a message in the chat to see the comparison")
            
        # Historical similarity data visualization
        st.subheader("Similarity Trend Analysis")
        
        # Display trend visualization if we have data
        if 'similarity_history' not in st.session_state or len(st.session_state.similarity_history) < 3:
            st.info("More data needed for trend analysis. Send at least 3 messages to see trends.")
        else:
            # Create DataFrame for visualization
            df = pd.DataFrame(st.session_state.similarity_history)
            
            # Create a simple line chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(df)), df['retrieval_similarity'], 'b-', label='Retrieval Similarity')
            ax.plot(range(len(df)), df['response_similarity'], 'g-', label='Response Similarity')
            ax.set_xlabel('Query Number')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Similarity Scores Over Time')
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True)
            st.pyplot(fig)
            
            # Show aggregated statistics
            st.subheader("Summary Statistics")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Avg Retrieval Similarity", f"{df['retrieval_similarity'].mean():.3f}")
                st.metric("Min Retrieval Similarity", f"{df['retrieval_similarity'].min():.3f}")
                st.metric("Max Retrieval Similarity", f"{df['retrieval_similarity'].max():.3f}")
            with stats_col2:
                st.metric("Avg Response Similarity", f"{df['response_similarity'].mean():.3f}")
                st.metric("Min Response Similarity", f"{df['response_similarity'].min():.3f}")
                st.metric("Max Response Similarity", f"{df['response_similarity'].max():.3f}")
                
# Add after imports
def check_required_files():
    required_files = [
        "labeled_qa.csv",
        "faculty.json",
        "general_info.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.write("Please make sure all required files are in the correct location:")
        for file in missing_files:
            st.write(f"- {file}")
        st.stop()

# Add this after check_required_files()
def verify_qa_data():
    try:
        qa_df = pd.read_csv("labeled_qa.csv")
        required_columns = ['Category', 'Question', 'Answer']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in qa_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in labeled_qa.csv: {', '.join(missing_columns)}")
            st.stop()
            
        # Check if there's data
        if len(qa_df) == 0:
            st.error("labeled_qa.csv is empty")
            st.stop()
            
        # st.success(f"Successfully loaded {len(qa_df)} QA pairs") DEBUGGING
        return qa_df
    except Exception as e:
        st.error(f"Error reading labeled_qa.csv: {str(e)}")
        st.stop()

# Add this call after check_required_files()
qa_df = verify_qa_data()

if __name__ == "__main__":
    main()