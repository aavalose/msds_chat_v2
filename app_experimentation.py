import streamlit as st
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import json
import os
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

# Handle missing API key safely
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please configure it in your Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB client
@st.cache_resource
def init_chroma():
    chroma_client = chromadb.Client()
    # Use OpenAI's embedding function (you can change this to other embedding functions if needed)
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return chroma_client, embedding_function

chroma_client, embedding_function = init_chroma()

# Create a collection with the specified embedding function
qa_collection = chroma_client.create_collection(
    name="msds_program_qa",
    embedding_function=embedding_function,
    get_or_create=True
)

# Load QA data
qa_df = pd.read_csv("Questions_and_Answers.csv")

# Add data to the collection
try:
    qa_collection.upsert(
        ids=[str(i) for i in qa_df.index.tolist()],  # Convert ids to strings
        documents=qa_df['Question'].tolist(),
        metadatas=qa_df[['Answer']].to_dict(orient='records')
    )
except Exception as e:
    st.error(f"Error adding data to collection: {str(e)}")

# Configure Gemini model
@st.cache_resource
def load_gemini_model():
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

gemini_model = load_gemini_model()

# Save conversation to a JSON file
def save_conversation(session_id, user_message, bot_response):
    os.makedirs("conversations", exist_ok=True)
    filename = "conversations/chat_history.json"
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []
        
        conversations.append({
            "session_id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_message": user_message,
            "bot_response": bot_response
        })
        
        with open(filename, 'w') as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        st.error(f"Error saving conversation: {str(e)}")

# Find the most similar question using ChromaDB
def find_most_similar_question(user_input, similarity_threshold=0.5):
    try:
        results = qa_collection.query(
            query_texts=[user_input],
            n_results=1
        )
        
        if results and results['documents'] and results['distances']:
            # ChromaDB returns cosine distance, convert to similarity
            similarity = 1 - results['distances'][0][0]  # Convert distance to similarity
            
            if similarity >= similarity_threshold:
                matched_question = results['documents'][0][0]
                matched_answer = results['metadatas'][0][0]['Answer']
                return matched_question, matched_answer, similarity
        
        return None, None, 0.0
    except Exception as e:
        st.error(f"Error finding similar question: {str(e)}")
        return None, None, 0.0

# Generate response using Gemini
def get_gemini_response(user_input, retrieved_question=None, retrieved_answer=None):
    try:
        if retrieved_question and retrieved_answer:
            prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS (Master of Science in Data Science) program. 
            A prospective or current student has asked: "{user_input}"
            
            I found a similar question in our database: "{retrieved_question}"
            With this official answer: "{retrieved_answer}"
            
            Please respond to the student's question in a natural, conversational way while:
            1. Maintaining accuracy of the official information
            2. Adapting the tone to be friendly and helpful
            3. Addressing their specific question directly
            4. Using clear and accessible language
            5. Adding a brief encouraging or helpful note if appropriate
            
            Remember to stay within the scope of the official answer while making it more conversational."""
        else:
            prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS (Master of Science in Data Science) program.
            
            A student has asked: "{user_input}"
            
            While I don't have a direct answer from our database for this specific question, please:
            1. Only respond if the question is clearly related to the USF MSDS program
            2. If it is related, politely explain that you don't have specific information about this aspect
            3. Suggest they contact the program office for accurate information
            4. Maintain a helpful and professional tone
            
            If the question is completely unrelated to the USF MSDS program, politely explain that you can only assist with MSDS program-related questions."""

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
        
    matched_question, matched_answer, similarity = find_most_similar_question(user_input)
    st.session_state.debug_similarity = similarity
    
    if matched_question and matched_answer:
        st.session_state.debug_matched_question = matched_question
        st.session_state.debug_matched_answer = matched_answer
        return get_gemini_response(user_input, matched_question, matched_answer)
    
    return "I'm sorry, but I can only answer questions related to the University of San Francisco's MSDS program."

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state variables
    for key in ['debug_matched_question', 'debug_matched_answer', 'debug_similarity', 'chat_history', 'session_id']:
        if key not in st.session_state:
            st.session_state[key] = "" if key != 'chat_history' else []
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
                    bot_response = get_gemini_response(q, matched_question, matched_answer) if matched_question else "I'm sorry, but I can only answer questions related to the USF MSDS program."
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
        
        # Display chat history in reverse chronological order
        for message in reversed(st.session_state.chat_history):
            if message["role"] == "user":
                st.write("🧑 **You:**")
                st.write(message["content"])
            else:
                st.write("🤖 **Assistant:**")
                st.write(message["content"])
            st.write("---")  # Add a separator between messages

if __name__ == "__main__":
    main()