import streamlit as st
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login, HfApi

# Handle missing API key safely
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", None)
if HF_API_KEY:
    login(HF_API_KEY)  # Authenticate

# Load CSV data and create embeddings
@st.cache_resource
def load_qa_data():
    qa_df = pd.read_csv("Questions_and_Answers.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    question_embeddings = embedder.encode(qa_df['Question'].tolist())
    return qa_df, embedder, question_embeddings

qa_df, embedder, question_embeddings = load_qa_data()

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLaMA model
@st.cache_resource
def load_llama_model():
    model_name = "meta-llama/Llama-3.2-1B"  # Updated name
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_KEY)
    
    # Adjust model loading based on GPU availability
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {"torch_dtype": torch.float32}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_API_KEY,
        device_map="auto" if device == "cuda" else None,
        **model_kwargs
    ).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

llama_model, llama_tokenizer = load_llama_model()

# Save conversation to a JSON file
def save_conversation(session_id, user_message, bot_response):
    os.makedirs("conversations", exist_ok=True)
    filename = f"conversations/{session_id}.json"
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []
        
        conversations.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_message": user_message,
            "bot_response": bot_response
        })
        
        with open(filename, 'w') as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        st.error(f"Error saving conversation: {str(e)}")

# Find the most similar question using cosine similarity
def find_most_similar_question(user_input, similarity_threshold=0.5):
    try:
        user_embedding = embedder.encode([user_input])[0]
        similarities = np.dot(question_embeddings, user_embedding) / (
            np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(user_embedding)
        )
        
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        
        if max_similarity >= similarity_threshold:
            return qa_df.iloc[max_sim_idx]['Question'], qa_df.iloc[max_sim_idx]['Answer'], max_similarity
        return None, None, max_similarity
    except Exception as e:
        st.error(f"Error finding similar question: {str(e)}")
        return None, None, 0.0

# Generate response using LLaMA
def get_llama_response(user_input, retrieved_question=None, retrieved_answer=None):
    try:
        if retrieved_question and retrieved_answer:
            context = f"""
            A student has asked: "{user_input}"
            The most relevant question from the database is: "{retrieved_question}"
            The official answer is: "{retrieved_answer}"
            Your task is to **restate this answer clearly and concisely** while keeping the original meaning intact.
            - **Do not invent** new information.
            - **Do not include unrelated details**.
            - Ensure the answer is precise and professional.
            Final Answer:
            """
        else:
            context = f"""
            The student has asked: "{user_input}"
            No relevant answer was found. Respond only if the question is related to the USF MSDS program.
            Final Answer:
            """

        inputs = llama_tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            output = llama_model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,  
                max_new_tokens=150,  # Increased for more complete responses
                no_repeat_ngram_size=3,  
                temperature=0.3,  
                top_p=0.8,        
                do_sample=True,
                pad_token_id=llama_tokenizer.pad_token_id
            )
        
        response = llama_tokenizer.decode(output[0], skip_special_tokens=True)
        
        if "Final Answer:" in response:
            response = response.split("Final Answer:")[1].strip()
        
        return response
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
        return get_llama_response(user_input, matched_question, matched_answer)
    
    return "I'm sorry, but I can only answer questions related to the University of San Francisco's MSDS program."

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state variables
    for key in ['debug_matched_question', 'debug_matched_answer', 'debug_llama_response', 'debug_similarity', 'chat_history', 'session_id']:
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
                    bot_response = get_llama_response(q, matched_question, matched_answer) if matched_question else "I'm sorry, but I can only answer questions related to the USF MSDS program."
                    st.session_state.chat_history.append(("You", q))
                    st.session_state.chat_history.append(("Bot", bot_response))
                    save_conversation(st.session_state.session_id, q, bot_response)
        
        st.subheader("Ask me about USF's MSDS program")
        user_message = st.text_input("Type your question here:", key="user_input")
        
        if st.button("Send", key="send_button") and user_message:
            with st.spinner("Thinking..."):
                bot_response = get_bot_response(user_message)
                st.session_state.chat_history.append(("You", user_message))
                st.session_state.chat_history.append(("Bot", bot_response))
                save_conversation(st.session_state.session_id, user_message, bot_response)
        
        # Display chat history in reverse chronological order
        for role, message in reversed(st.session_state.chat_history):
            st.write(f"**{role}:** {message}")

if __name__ == "__main__":
    main()