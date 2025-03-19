import streamlit as st
import numpy as np
from datetime import datetime
import pandas as pd
import json
import os
import sys
import google.generativeai as genai
from pymongo import MongoClient

# Configure sqlite3 to use pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
    # Create a persistent directory for ChromaDB
    os.makedirs("chroma_db", exist_ok=True)
    
    # Initialize the client with persistence
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    
    # Use default embedding function
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return chroma_client, embedding_function

chroma_client, embedding_function = init_chroma()

# Create a collection with the specified embedding function
try:
    qa_collection = chroma_client.create_collection(
        name="msds_program_qa",
        embedding_function=embedding_function,
        get_or_create=True
    )
    
    # Load QA data
    try:
        qa_df = pd.read_csv("Questions_and_Answers.csv")
        
        # Add data to the collection
        try:
            # Get all existing IDs
            existing_ids = qa_collection.get()["ids"]
            if existing_ids:
                # Delete existing documents if any
                qa_collection.delete(ids=existing_ids)
            
            # Then add new documents
            qa_collection.upsert(
                ids=[str(i) for i in qa_df.index.tolist()],  # Convert ids to strings
                documents=qa_df['Question'].tolist(),
                metadatas=qa_df[['Answer']].to_dict(orient='records')
            )
        except Exception as e:
            st.error(f"Error adding data to collection: {str(e)}")
    except Exception as e:
        st.error(f"Error loading Questions_and_Answers.csv: {str(e)}")
except Exception as e:
    st.error(f"Error creating ChromaDB collection: {str(e)}")
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

# Modify save_conversation to handle None collection
def save_conversation(session_id, user_message, bot_response):
    if conversations_collection is None:
        st.error("MongoDB connection not available")
        return
        
    try:
        conversation = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "user_name": st.session_state.user_name,
            "user_email": st.session_state.user_email,
            "user_message": user_message,
            "bot_response": bot_response
        }
        conversations_collection.insert_one(conversation)
    except Exception as e:
        st.error(f"Error saving conversation to MongoDB: {str(e)}")

# Find the most similar question using ChromaDB
def find_most_similar_question(user_input, similarity_threshold=0.45):
    try:
        # Add error handling for empty collection
        if qa_collection.count() == 0:
            return None, None, 0.0
            
        results = qa_collection.query(
            query_texts=[user_input],
            n_results=1
        )
        
        if results['documents'][0]:
            similarity = 1 - results['distances'][0][0]  # Convert distance to similarity
            
            if similarity >= similarity_threshold:
                matched_question = results['documents'][0][0]
                matched_answer = results['metadatas'][0][0]['Answer']
                return matched_question, matched_answer, similarity
        
        return None, None, 0.0
    except Exception:
        # Just return None values without showing any error
        return None, None, 0.0

# Generate response using Gemini
def get_gemini_response(user_input, retrieved_question=None, retrieved_answer=None):
    try:
        # Check if the query is about faculty
        if "faculty" in user_input.lower() or "professor" in user_input.lower() or "instructor" in user_input.lower():
            prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS (Master of Science in Data Science) program.
            
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
        else:
            # Load general information
            general_info = open('general_info.txt', 'r').read()
            
            if retrieved_question and retrieved_answer:
                prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS (Master of Science in Data Science) program. 
                A prospective or current student has asked: "{user_input}"
                
                I found a similar question in our database: "{retrieved_question}"
                With this official answer: "{retrieved_answer}"
                
                I also have this general information about the program:
                ```
                {general_info}
                ```
                
                Please respond to the student's question in a natural, conversational way while:
                1. Primarily using the matched question/answer as your main source of information
                2. Supplementing with relevant general information if helpful
                3. Maintaining accuracy of the official information
                4. Using a friendly and helpful tone
                5. Addressing their specific question directly
                6. Using clear and accessible language
                
                If the student's question isn't fully addressed by the matched answer, you may draw from the general information to provide a more complete response."""
            else:
                prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS (Master of Science in Data Science) program.
                
                A student has asked: "{user_input}"
                
                Please use this general information about the program to help answer their question:
                ```
                {general_info}
                ```
                
                Please:
                1. If the answer can be found in the general information, provide a helpful and accurate response
                2. Only respond to questions related to the USF MSDS program
                3. If you don't have enough information to fully answer their question:
                   - Share what relevant information you do have
                   - Acknowledge what specific aspects you don't have information about
                   - Suggest they contact the program office for those specific details
                4. Maintain a helpful and professional tone
                5. Be clear about what you know vs. what you're unsure about"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."

# Get bot response
def get_bot_response(user_input):
    if not user_input.strip():
        return "Please enter a question."
    
    # First try to find a similar question
    matched_question, matched_answer, similarity = find_most_similar_question(user_input)
    st.session_state.debug_similarity = similarity
    
    # Store matched Q&A in session state for debugging
    st.session_state.debug_matched_question = matched_question if matched_question else ""
    st.session_state.debug_matched_answer = matched_answer if matched_answer else ""
    
    # Generate response using Gemini, passing matched Q&A if found
    return get_gemini_response(user_input, matched_question, matched_answer)

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state variables
    for key in ['debug_matched_question', 'debug_matched_answer', 'debug_similarity', 'chat_history', 'session_id', 'user_name', 'user_email']:
        if key not in st.session_state:
            st.session_state[key] = "" if key != 'chat_history' else []
            if key == 'debug_similarity':
                st.session_state[key] = 0.0
            elif key == 'session_id':
                st.session_state[key] = datetime.now().strftime("%Y%m%d-%H%M%S")

    tab1, tab2, tab3 = st.tabs(["Chat", "About", "Debug"])

    with tab1:
        # Add user information collection at the top of the chat tab
        if not st.session_state.user_name or not st.session_state.user_email:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_name = st.text_input("Please enter your name:", key="name_input")
            with col2:
                st.session_state.user_email = st.text_input("Please enter your email:", key="email_input")
            
            if not st.session_state.user_name or not st.session_state.user_email:
                st.warning("Please provide both your name and email to continue.")
                st.stop()

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
                    save_conversation(st.session_state.session_id, q, bot_response)
        
        st.subheader("Ask me about USF's MSDS program")
        user_message = st.text_input("Type your question here:", key="user_input")
        
        if st.button("Send", key="send_button") and user_message:
            with st.spinner("Thinking..."):
                bot_response = get_bot_response(user_message)
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                save_conversation(st.session_state.session_id, user_message, bot_response)
        
        # Get chat history pairs in reverse order (newest first)
        chat_pairs = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_msg = st.session_state.chat_history[i]
                bot_msg = st.session_state.chat_history[i + 1]
                chat_pairs.append((user_msg, bot_msg))

        # Display newest messages first
        for user_msg, bot_msg in reversed(chat_pairs):
            st.write("🧑 **You:**")
            st.write(user_msg["content"])
            st.write("🤖 **Assistant:**")
            st.write(bot_msg["content"])
            st.write("---")

if __name__ == "__main__":
    main()