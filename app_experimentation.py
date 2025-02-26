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

HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
login(HF_API_KEY)  # Authenticate

# Load CSV data and create embeddings
@st.cache_resource
def load_qa_data():
    # Load Questions and Answers from CSV
    qa_df = pd.read_csv("Questions_and_Answers.csv")
    
    # Initialize sentence embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings for all questions
    question_embeddings = embedder.encode(qa_df['Question'].tolist())
    
    return qa_df, embedder, question_embeddings

qa_df, embedder, question_embeddings = load_qa_data()

# Load LLaMA model
@st.cache_resource
def load_llama_model():
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

llama_model, llama_tokenizer = load_llama_model()

# Save conversation to a JSON file
def save_conversation(session_id, user_message, bot_response):
    # Create conversations directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    # Create a filename based on session_id
    filename = f"conversations/{session_id}.json"
    
    # Load existing conversations if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            conversations = json.load(f)
    else:
        conversations = []
    
    # Add new conversation
    conversations.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_message": user_message,
        "bot_response": bot_response
    })
    
    # Save updated conversations
    with open(filename, 'w') as f:
        json.dump(conversations, f, indent=2)

# Find the most similar question using cosine similarity
def find_most_similar_question(user_input, similarity_threshold=0.5):
    # Encode user input
    user_embedding = embedder.encode([user_input])[0]
    
    # Calculate cosine similarity with all question embeddings
    similarities = np.dot(question_embeddings, user_embedding) / (
        np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )
    
    # Find the most similar question
    max_sim_idx = np.argmax(similarities)
    max_similarity = similarities[max_sim_idx]
    
    if max_similarity >= similarity_threshold:
        matched_question = qa_df.iloc[max_sim_idx]['Question']
        matched_answer = qa_df.iloc[max_sim_idx]['Answer']
        return matched_question, matched_answer, max_similarity
    else:
        return None, None, max_similarity  # No relevant question found

# Generate a response using LLaMA with improved prompting
def get_llama_response(user_input, retrieved_question=None, retrieved_answer=None):
    if retrieved_question and retrieved_answer:
        context = f"""
        A student has asked: "{user_input}"
        
        The most relevant question from the database is: "{retrieved_question}"
        The official answer is: "{retrieved_answer}"
        
        Your task is to **restate this answer clearly and concisely** while keeping the original meaning intact.
        - **Do not invent** new information.
        - **Do not include unrelated details**.
        - Ensure the answer is precise and professional.

        Provide only the corrected and well-structured answer below:

        Final Answer:
        """
    else:
        context = f"""
        The student has asked: "{user_input}"
        
        No relevant answer was found. Respond only if the question is related to the USF MSDS program.
        
        If it is not related, politely inform the user that you can only provide information about the USF MSDS program.

        Final Answer:
        """

    inputs = llama_tokenizer(context, return_tensors="pt", padding=True, truncation=True).to("cuda")
    output = llama_model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,  
        max_new_tokens=100,  # Keeping the response brief to prevent hallucination
        no_repeat_ngram_size=3,  
        temperature=0.3,  # Lower temperature to reduce randomness
        top_p=0.8,        
        do_sample=True    
    )
    response = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the final answer if needed
    if "Final Answer:" in response:
        response = response.split("Final Answer:")[1].strip()
    
    return response

# Get bot response with improved quality
def get_bot_response(user_input):
    matched_question, matched_answer, similarity = find_most_similar_question(user_input)
    
    st.session_state.debug_similarity = similarity
    
    # If a relevant match is found, refine the answer with LLaMA
    if matched_question and matched_answer:
        st.session_state.debug_matched_question = matched_question
        st.session_state.debug_matched_answer = matched_answer
        
        return get_llama_response(user_input, matched_question, matched_answer)
    
    # If no match is found, return a polite message without calling LLaMA
    return "I'm sorry, but I can only answer questions related to the University of San Francisco's MSDS program. Please ask about admissions, curriculum, faculty, or other program details."

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize debug state variables if they don't exist
    if 'debug_matched_question' not in st.session_state:
        st.session_state.debug_matched_question = ""
    if 'debug_matched_answer' not in st.session_state:
        st.session_state.debug_matched_answer = ""
    if 'debug_llama_response' not in st.session_state:
        st.session_state.debug_llama_response = ""
    if 'debug_similarity' not in st.session_state:
        st.session_state.debug_similarity = 0.0
    
    # Use tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Chat", "About", "Debug"])
    
    with tab1:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Session management in the sidebar
        with st.sidebar:
            st.subheader("Session Management")
            st.write(f"Current Session ID: {st.session_state.session_id}")
            
            if st.button("Start New Session"):
                st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                st.session_state.chat_history = []
                st.rerun()  # Updated from experimental_rerun()
            
            # Example questions to help users
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
              if st.button(q):
                  # Ensure the question goes through the retrieval process
                  matched_question, matched_answer, similarity = find_most_similar_question(q)

                  if matched_question and matched_answer:
                      bot_response = get_llama_response(q, matched_question, matched_answer)
                  else:
                      bot_response = "I'm sorry, but I can only answer questions related to the University of San Francisco's MSDS program."

                  st.session_state.chat_history.append(("You", q))
                  st.session_state.chat_history.append(("Bot", bot_response))
                  save_conversation(st.session_state.session_id, q, bot_response)
        
        # Chat interface
        st.subheader("Ask me about USF's MSDS program")
        user_message = st.text_input("Type your question here:", key="user_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("Send", use_container_width=True)
        
        with col2:
            clear_button = st.button("Clear Chat", use_container_width=True)
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()  # Updated from experimental_rerun()
        
        if send_button and user_message:
            with st.spinner("Thinking..."):
                bot_response = get_bot_response(user_message)
                
                st.session_state.chat_history.append(("You", user_message))
                st.session_state.chat_history.append(("Bot", bot_response))
                
                save_conversation(st.session_state.session_id, user_message, bot_response)
        
        # Display chat history
        st.subheader("Conversation:")
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.write(f"👤 **You:** {message}")
            else:
                st.markdown(f"🤖 **Bot:** {message}")
    
    with tab2:
        st.header("About this Chatbot")
        st.write("""
        This chatbot is designed to provide information specifically about the University of San Francisco's Master of Science in Data Science (MSDS) program.
        
        ### Features:
        - Answers questions about admissions, curriculum, faculty, and other MSDS program details
        - Uses a combination of pre-defined answers and AI-generated responses
        - Focuses exclusively on USF MSDS program information
        
        ### How it works:
        1. Your question is compared to a database of common MSDS program questions
        2. If a match is found, you receive a curated answer
        3. If no match is found, an AI model generates a response
        4. The chatbot will politely decline to answer questions unrelated to the USF MSDS program
        
        ### Data sources:
        The information provided is based on the official USF MSDS program details and commonly asked questions.
        """)
    
    with tab3:
        st.header("Debug Information")
        st.subheader("Question Match Information:")
        st.write(f"Similarity score: {st.session_state.debug_similarity:.4f}")
        st.write(f"Matched question: {st.session_state.debug_matched_question}")
        st.write(f"Matched answer: {st.session_state.debug_matched_answer}")
        
        st.subheader("LLaMA Generated Response:")
        st.write(st.session_state.debug_llama_response)
        
        # CSV Data Preview
        st.subheader("CSV Data Preview")
        if st.button("Show QA Data Sample"):
            st.dataframe(qa_df.head(10))
            st.info(f"Total Q&A pairs in dataset: {len(qa_df)}")

if __name__ == "__main__":
    main()