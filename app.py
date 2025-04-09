import streamlit as st
from datetime import datetime
from src.database.mongodb import init_mongodb, save_conversation, update_feedback
from src.database.chromadb import init_chroma, load_and_index_json_data
from src.llm.gemini import load_gemini_model, get_gemini_response
from src.retrieval.qa_retrieval import find_most_similar_question
from src.utils.preprocessing import preprocess_query
from src.utils.similarity import calculate_cosine_similarity

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

    # Create tabs
    tab1, tab2 = st.tabs(["Chat", "About"])

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
                if st.button(q, key=f"btn_{q[:20]}"):
                    matched_question, matched_answer, similarity = find_most_similar_question(q)
                    bot_response = get_gemini_response(q, matched_question, matched_answer)
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                    save_conversation(st.session_state.session_id, q, bot_response, 0.0)
        
        st.subheader("Ask me about USF's MSDS program")
        user_message = st.text_input("Type your question here:", key="user_input")
        
        if st.button("Send", key="send_button") and user_message:
            with st.spinner("Thinking..."):
                start_time = datetime.now()
                matched_question, matched_answer, similarity = find_most_similar_question(user_message)
                bot_response = get_gemini_response(user_message, matched_question, matched_answer)
                response_time = (datetime.now() - start_time).total_seconds()
                
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                
                conversation_id = save_conversation(
                    st.session_state.session_id, 
                    user_message, 
                    bot_response,
                    response_time,
                    response_similarity=st.session_state.debug_response_similarity
                )
                if conversation_id:
                    if 'conversation_ids' not in st.session_state:
                        st.session_state.conversation_ids = []
                    st.session_state.conversation_ids.append(conversation_id)
        
        # Display chat history
        chat_pairs = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_msg = st.session_state.chat_history[i]
                bot_msg = st.session_state.chat_history[i + 1]
                chat_pairs.append((user_msg, bot_msg))

        for i, (user_msg, bot_msg) in enumerate(reversed(chat_pairs)):
            # User message
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

            # Bot message
            bot_container = st.container()
            with bot_container:
                col1, col2 = st.columns([1, 6])
                with col1:
                    st.write("🤖")
                with col2:
                    sanitized_content = bot_msg["content"]
                    problematic_tags = ['</div>', '<div>', '</span>', '<span>']
                    for tag in problematic_tags:
                        sanitized_content = sanitized_content.replace(tag, tag.replace('<', '&lt;').replace('>', '&gt;'))
                    
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
                            {sanitized_content}
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