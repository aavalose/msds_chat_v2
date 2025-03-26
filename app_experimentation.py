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
@st.cache_resource
def init_qa_collection(_chroma_client, _embedding_function):
    try:
        # Try to get existing collection first
        try:
            qa_collection = _chroma_client.get_collection(
                name="msds_program_qa",
                embedding_function=_embedding_function
            )
            st.success("Successfully connected to existing QA collection")
        except:
            # If collection doesn't exist, create it and load data
            qa_collection = _chroma_client.create_collection(
                name="msds_program_qa",
                embedding_function=_embedding_function
            )
            st.info("Created new QA collection")

            # Load QA data
            try:
                qa_df = pd.read_csv("labeled_qa.csv")
                
                # Add data to the collection
                qa_collection.add(
                    ids=[str(i) for i in qa_df.index.tolist()],
                    documents=qa_df['Question'].tolist(),
                    metadatas=qa_df[['Answer', 'Category']].to_dict(orient='records')
                )
                st.success("Successfully loaded QA data")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                raise e

        return qa_collection
    except Exception as e:
        st.error(f"Error initializing QA collection: {str(e)}")
        raise e

# Initialize ChromaDB and collection
try:
    chroma_client, embedding_function = init_chroma()
    qa_collection = init_qa_collection(chroma_client, embedding_function)
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

# Modify save_conversation to handle None collection and store metrics
def save_conversation(session_id, user_message, bot_response, response_time, metrics=None):
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
            "response_time_seconds": response_time
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

# Find the most similar question using ChromaDB
def find_most_similar_question(user_input, similarity_threshold=0.3):
    try:
        if qa_collection.count() == 0:
            return None, None, 0.0
        
        # Preprocess the user input and get category
        processed_input, category = preprocess_query(user_input)
        
        # Query with filter for matching category
        results = qa_collection.query(
            query_texts=[processed_input],
            n_results=5,
            where={"category": category} if category and category != "Other" else None
        )
        
        if not results['documents'][0]:
            return None, None, 0.0
        
        # Find the best match among the top results
        best_similarity = 0.0
        best_question = None
        best_answer = None
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write("Top matches:")
            st.write(f"Category: {category}")
            for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                sim = 1 - dist
                st.write(f"{i+1}. Question: {doc}")
                st.write(f"   Similarity: {sim:.3f}")
                st.write(f"   Category: {results['metadatas'][0][i].get('category', 'Unknown')}")
        
        for i, distance in enumerate(results['distances'][0]):
            similarity = 1 - distance
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_question = results['documents'][0][i]
                best_answer = results['metadatas'][0][i]['Answer']
        
        return best_question, best_answer, best_similarity
            
    except Exception as e:
        st.error(f"Error in find_most_similar_question: {str(e)}")
        return None, None, 0.0

# Enhance the preprocess_query function
def preprocess_query(query):
    """Normalize and expand common variations in queries"""
    # Initialize processed_query with the normalized input
    processed_query = query.lower().strip()
   
    # Categorize the query using Gemini
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Categorize this question into exactly ONE of the following categories:
        - Admissions: Questions about getting into the program, requirements, deadlines
        - Application: Questions about the application process, documents needed
        - Career: Questions about job prospects, career paths, industry outcomes
        - Classes: Questions about specific courses, curriculum, class schedule, workload
        - Faculty: Questions about professors, instructors, program directors, staff
        - Graduation: Questions about graduation requirements, ceremonies, timelines
        - Program: Questions about program structure, cohorts, learning outcomes
        - Salary: Questions about expected income, salary ranges, compensation
        - Time: Questions about program duration, time commitment, schedules
        - Tuition: Questions about costs, financial aid, scholarships, payment plans
        - Other: Questions that don't fit into the above categories
        
        Examples:
        Question: "What GRE score do I need?" -> Admissions
        Question: "How much is tuition per semester?" -> Tuition
        Question: "Who teaches the machine learning course?" -> Faculty
        Question: "How long does it take to complete the program?" -> Time
        Question: "What programming languages will I learn?" -> Program
        Your question: "{query}"
        
        Return only the category name, nothing else."""
        
        response = model.generate_content(prompt)
        category = response.text.strip()
        
        # Add the category as metadata
        processed_query = f"{processed_query} [category:{category}]"
    except Exception as e:
        # If categorization fails, set category to "Other"
        category = "Other"
        if st.session_state.get('debug_mode', False):
            st.error(f"Error categorizing query: {str(e)}")
    
    return processed_query, category

# Generate response using Gemini
def get_gemini_response(user_input, retrieved_question=None, retrieved_answer=None):
    try:
        # Load general information
        general_info = open('general_info.txt', 'r').read()
        
        # Process the query to get category
        processed_query, category = preprocess_query(user_input)
        
        # Check if the query is about faculty
        if "faculty" in user_input.lower() or "professor" in user_input.lower() or "instructor" in user_input.lower() or category == "Faculty":
            prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
            
            A student has asked about faculty: "{user_input}"
            
            Please use the following faculty information to answer their question:
            
            ```
            {open('faculty.json', 'r').read()}
            ```
            
            Please respond in a natural, conversational way while:
            1. Providing accurate information about the faculty members
            2. Being friendly and helpful
            3. Addressing their specific question directly
            4. Using clear and accessible language
            """
        # Check if the query is about courses
        elif "course" in user_input.lower() or "class" in user_input.lower() or "curriculum" in user_input.lower() or category == "Classes":
            prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
            
            A student has asked about courses: "{user_input}"
            
            Please use the following course information to answer their question:
            
            ```
            {open('courses.json', 'r').read()}
            ```
            
            Please respond in a natural, conversational way while:
            1. Providing accurate information about the MSDS program courses and curriculum
            2. Being friendly and helpful
            3. Addressing their specific question directly
            4. Using clear and accessible language
            """
        else:
            if retrieved_question and retrieved_answer and st.session_state.debug_similarity >= 0.3:
                prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
                
                User question: "{user_input}"
                Category: {category}
                
                Official reference:
                Question: "{retrieved_question}"
                Answer: "{retrieved_answer}"
                
                Instructions:
                1. If the official answer contains specific facts, numbers, or requirements, preserve them exactly
                2. Focus on answering the user's specific question
                3. Use a conversational tone while maintaining accuracy
                4. If any information is missing or unclear, acknowledge it
                
                Additional context:
                {general_info}
                
                Please provide your response:"""
            else:
                prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
                
                User question: "{user_input}"
                Category: {category}
                
                Please use this general information to help answer their question:
                ```
                {general_info}
                ```
                
                Instructions:
                1. If the answer can be found in the general information, provide a helpful and accurate response
                2. Only respond to questions related to the USF MSDS program
                3. If you don't have enough information:
                   - Share what relevant information you do have
                   - Acknowledge what you don't know
                   - Suggest contacting the program office for those details
                4. Use a helpful and professional tone
                5. Be clear about what you know vs. what you're unsure about"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."




# Get bot response (modified to include metrics)
def get_bot_response(user_input):
    if not user_input.strip():
        return "Please enter a question.", None
    
    # Process the query to get category
    processed_query, category = preprocess_query(user_input)
    
    # First try to find a similar question
    matched_question, matched_answer, similarity = find_most_similar_question(user_input)
    
    # Debug information
    st.session_state.debug_similarity = similarity
    st.session_state.debug_matched_question = matched_question if matched_question else "No match found"
    st.session_state.debug_matched_answer = matched_answer if matched_answer else "No answer found"
    st.session_state.debug_category = category if category else "No category found"
    
    # Add debug output
    if st.session_state.get('debug_mode', False):
        st.write("Debug Info:")
        st.write(f"Category: {category}")
        st.write(f"Similarity Score: {similarity:.3f}")
        st.write(f"Matched Question: {matched_question}")
        st.write(f"Matched Answer: {matched_answer}")
    
    # Generate response using Gemini, passing matched Q&A if found
    bot_response = get_gemini_response(user_input, matched_question, matched_answer)
    
    # Return the response
    return bot_response

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state variables
    for key in ['debug_matched_question', 'debug_matched_answer', 'debug_similarity', 
                'chat_history', 'session_id', 'conversation_ids', 'debug_category']:
        if key not in st.session_state:
            st.session_state[key] = "" if key not in ['chat_history', 'conversation_ids'] else []
            if key == 'debug_similarity':
                st.session_state[key] = 0.0
            elif key == 'session_id':
                st.session_state[key] = datetime.now().strftime("%Y%m%d-%H%M%S")

    tab1, tab2, tab3 = st.tabs(["Chat", "About", "Debug"])

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
                    response_time
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
            st.write("🧑 **You:**")
            st.write(user_msg["content"])
            st.write("🤖 **Assistant:**")
            st.write(bot_msg["content"])
            
            # Add feedback buttons
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
            st.write("---")

    with tab3:
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if st.session_state.debug_mode:
            st.write("Last Query Debug Info:")
            st.write(f"Category: {st.session_state.debug_category}")
            st.write(f"Similarity Score: {st.session_state.debug_similarity:.3f}")
            st.write(f"Matched Question: {st.session_state.debug_matched_question}")
            st.write(f"Matched Answer: {st.session_state.debug_matched_answer}")

# Add after imports
def check_required_files():
    required_files = [
        "Questions_and_Answers.csv",
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
        qa_df = pd.read_csv("Questions_and_Answers.csv")
        required_columns = ['Question', 'Answer']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in qa_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in Questions_and_Answers.csv: {', '.join(missing_columns)}")
            st.stop()
            
        # Check if there's data
        if len(qa_df) == 0:
            st.error("Questions_and_Answers.csv is empty")
            st.stop()
            
        st.success(f"Successfully loaded {len(qa_df)} QA pairs")
        return qa_df
    except Exception as e:
        st.error(f"Error reading Questions_and_Answers.csv: {str(e)}")
        st.stop()

# Add this call after check_required_files()
qa_df = verify_qa_data()

if __name__ == "__main__":
    main()