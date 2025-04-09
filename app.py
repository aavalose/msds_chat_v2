import streamlit as st
from datetime import datetime
from src.database.mongodb import init_mongodb, save_conversation, update_feedback, save_metrics
from src.database.chromadb import init_chroma, load_and_index_json_data
from src.llm.gemini import load_gemini_model, get_gemini_response
from src.retrieval.qa_retrieval import find_most_similar_question
from src.utils.preprocessing import preprocess_query
from src.utils.similarity import calculate_cosine_similarity
from src.utils.similarity_analysis import analyze_conversation
import pandas as pd

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if 'conversation_ids' not in st.session_state:
        st.session_state.conversation_ids = []
    if 'debug_similarity' not in st.session_state:
        st.session_state.debug_similarity = 0.0
    if 'debug_response_similarity' not in st.session_state:
        st.session_state.debug_response_similarity = 0.0
    if 'debug_matched_question' not in st.session_state:
        st.session_state.debug_matched_question = ""
    if 'debug_matched_answer' not in st.session_state:
        st.session_state.debug_matched_answer = ""
    if 'debug_category' not in st.session_state:
        st.session_state.debug_category = ""

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize ChromaDB and collection
    try:
        chroma_client, embedding_function = init_chroma()
        qa_collection = load_and_index_json_data(chroma_client, embedding_function)
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {str(e)}")
        st.stop()

    # Initialize MongoDB
    conversations_collection = init_mongodb()
    if conversations_collection is None:
        st.error("Failed to initialize MongoDB")
        st.stop()

    # Initialize Gemini model
    gemini_model = load_gemini_model()

    # Load labeled QA data for similarity analysis
    try:
        labeled_qa = pd.read_csv("data/Questions_and_Answers.csv")
    except Exception as e:
        st.error(f"Failed to load labeled QA data: {str(e)}")
        labeled_qa = None

    # Create tabs
    tab1, tab2 = st.tabs(["Chat", "About"])

    with tab1:
        # Create a container for the chat interface
        chat_container = st.container()
        
        # Sidebar with session management and example questions
        with st.sidebar:
            st.subheader("Session Management")
            st.write(f"Current Session ID: {st.session_state.session_id}")

            if st.button("Start New Session"):
                st.session_state.messages = []
                st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
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
                if st.button(q, key=f"btn_{q[:20]}"):
                    matched_question, matched_answer, similarity = find_most_similar_question(q)
                    bot_response = get_gemini_response(q, matched_question, matched_answer)
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                    # Calculate similarity metrics if labeled QA data is available
                    if labeled_qa is not None:
                        # Find the most similar question in labeled QA data
                        qa_similarity = labeled_qa['Question'].apply(
                            lambda x: calculate_cosine_similarity(q, x)
                        )
                        most_similar_idx = qa_similarity.idxmax()
                        expert_response = labeled_qa.loc[most_similar_idx, 'Answer']
                        
                        # Calculate metrics
                        metrics = analyze_conversation(bot_response, q, expert_response)
                        
                        # Save metrics to separate collection
                        save_metrics(metrics)
                    
                    save_conversation(st.session_state.session_id, q, bot_response, 0.0)
        
        # Chat input at the bottom
        prompt = st.chat_input("Ask me anything about the MSDS program...")
        
        # Display chat messages in the container
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Add feedback buttons and text input for the last bot response
                    if message["role"] == "assistant" and message == st.session_state.messages[-1]:
                        col1, col2, col3 = st.columns([1, 1, 3])
                        with col1:
                            if st.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}"):
                                if st.session_state.conversation_ids:
                                    update_feedback(st.session_state.conversation_ids[-1], "positive")
                                    st.success("Thank you for your feedback!")
                        with col2:
                            if st.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}"):
                                if st.session_state.conversation_ids:
                                    update_feedback(st.session_state.conversation_ids[-1], "negative")
                                    st.success("Thank you for your feedback!")
                        with col3:
                            feedback_text = st.text_input("Additional feedback (optional)", key=f"feedback_text_{len(st.session_state.messages)}")
                            if feedback_text:
                                if st.session_state.conversation_ids:
                                    update_feedback(st.session_state.conversation_ids[-1], feedback_text)
                                    st.success("Thank you for your detailed feedback!")
        
        # Handle new messages
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get bot response
            with st.spinner("Thinking..."):
                start_time = datetime.now()
                matched_question, matched_answer, similarity = find_most_similar_question(prompt)
                response = get_gemini_response(prompt, matched_question, matched_answer)
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Calculate similarity metrics if labeled QA data is available
                if labeled_qa is not None:
                    # Find the most similar question in labeled QA data
                    qa_similarity = labeled_qa['Question'].apply(
                        lambda x: calculate_cosine_similarity(prompt, x)
                    )
                    most_similar_idx = qa_similarity.idxmax()
                    expert_response = labeled_qa.loc[most_similar_idx, 'Answer']
                    
                    # Calculate metrics
                    metrics = analyze_conversation(response, prompt, expert_response)
                    
                    # Save metrics to separate collection
                    save_metrics(metrics)
                
                # Save conversation to MongoDB
                conversation_id = save_conversation(
                    st.session_state.session_id,
                    prompt,
                    response,
                    response_time,
                    response_similarity=st.session_state.debug_response_similarity
                )
                if conversation_id:
                    st.session_state.conversation_ids.append(conversation_id)
            
            # Rerun to update the chat display
            st.rerun()
    
    # About tab
    with tab2:
        st.header("About this Chatbot")
        
        st.markdown("""
        ### USF MSDS Program Chatbot
        
        Welcome to the University of San Francisco's Master of Science in Data Science (MSDS) Program Chatbot. This intelligent assistant is designed to provide prospective and current students with accurate information about the MSDS program.
        
        ### Features
        
        - **Instant Answers**: Get immediate responses to your questions about admissions, curriculum, faculty, and more
        - **Smart Retrieval**: The chatbot uses advanced retrieval-augmented generation (RAG) to provide accurate program information
        - **Conversation Memory**: The bot remembers your previous questions in the current session for more natural conversations
        
        ### About the Developers
        
        This chatbot was developed by Sehej Singh and Arturo Avalos, graduate students in the USF MSDS program. We created this tool to help prospective students get quick and accurate answers to their questions about the program.
        
        Sehej and Arturo worked collaboratively on all aspects of this project, combining their expertise as machine learning engineer interns with years of experience in data science. Their joint efforts covered everything from designing the retrieval system and response generation to developing the user interface and backend integration.
        
        ### Technology
        
        This chatbot utilizes several advanced technologies:
        
        - **Google Gemini AI** for natural language understanding and generation
        - **ChromaDB** for vector storage and semantic search
        - **MongoDB** for conversation logging and analytics
        - **Streamlit** for the web interface
        
        ### Feedback
        
        We value your feedback! Please use the thumbs up/down buttons after each response to help us improve the chatbot.
        
        For more information about the USF MSDS program, visit [https://www.usfca.edu/arts-sciences/graduate-programs/data-science](https://www.usfca.edu/arts-sciences/graduate-programs/data-science)
        """)
        
        # Add images
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            try:
                st.image("images/usf.png", width=100)
                st.write("© University of San Francisco, 2025")
            except Exception:
                st.write("USF logo not found. Add it to your images folder.")
        
        with col2:
            try:
                st.image("images/sehej.jpeg", width=150)
                st.markdown("**Sehej Singh**")
            except Exception:
                st.write("Sehej's image not found. Add it to your images folder.")
                
        with col3:
            try:
                st.image("images/arturo.jpeg", width=150)
                st.markdown("**Arturo Avalos**")
            except Exception:
                st.write("Arturo's image not found. Add it to your images folder.")

if __name__ == "__main__":
    main() 