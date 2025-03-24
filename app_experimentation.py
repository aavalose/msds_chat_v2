import streamlit as st
from datetime import datetime
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import the modules
from src.models.gemini_model import GeminiModel
from src.database.chroma_client import ChromaDBClient
from src.database.mongo_client import MongoDBClient
from src.chat.chat_processor import ChatProcessor

# Initialize components
@st.cache_resource
def init_components():
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found in Streamlit secrets.")
            st.stop()
            
        gemini_model = GeminiModel(api_key)
        chroma_client = ChromaDBClient()
        mongo_client = MongoDBClient(st.secrets["MONGO_CONNECTION_STRING"])
        chat_processor = ChatProcessor(gemini_model, chroma_client, mongo_client)
        return chat_processor
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize chat processor
    chat_processor = init_components()
    
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if "conversation_ids" not in st.session_state:
        st.session_state.conversation_ids = []
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

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
                if st.button(q, key=f"btn_{q[:20]}"):
                    bot_response, conversation_id, debug_info = chat_processor.process_message(
                        q, st.session_state.session_id
                    )
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": bot_response}
                    ])
                    if conversation_id:
                        st.session_state.conversation_ids.append(conversation_id)
        
        st.subheader("Ask me about USF's MSDS program")
        user_message = st.text_input("Type your question here:", key="user_input")
        
        if st.button("Send", key="send_button") and user_message:
            with st.spinner("Thinking..."):
                bot_response, conversation_id, debug_info = chat_processor.process_message(
                    user_message, st.session_state.session_id
                )
                
                # Update session state
                st.session_state.chat_history.extend([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": bot_response}
                ])
                if conversation_id:
                    st.session_state.conversation_ids.append(conversation_id)
                
                # Store debug information
                if debug_info:
                    st.session_state.debug_similarity = debug_info.get('similarity', 0.0)
                    st.session_state.debug_matched_question = debug_info.get('matched_question', 'No match found')
                    st.session_state.debug_matched_answer = debug_info.get('matched_answer', 'No answer found')
        
        # Display chat history
        chat_pairs = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                chat_pairs.append((
                    st.session_state.chat_history[i],
                    st.session_state.chat_history[i + 1]
                ))

        # Display messages newest first
        for i, (user_msg, bot_msg) in enumerate(reversed(chat_pairs)):
            st.write("🧑 **You:**")
            st.write(user_msg["content"])
            st.write("🤖 **Assistant:**")
            st.write(bot_msg["content"])
            
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍", key=f"thumbs_up_{i}"):
                    if i < len(st.session_state.conversation_ids):
                        chat_processor.update_feedback(
                            st.session_state.conversation_ids[-(i+1)], 
                            "positive"
                        )
                        st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎", key=f"thumbs_down_{i}"):
                    if i < len(st.session_state.conversation_ids):
                        chat_processor.update_feedback(
                            st.session_state.conversation_ids[-(i+1)], 
                            "negative"
                        )
                        st.success("Thank you for your feedback!")
            st.write("---")

    with tab3:
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if st.session_state.debug_mode:
            st.write("Last Query Debug Info:")
            st.write(f"Similarity Score: {st.session_state.debug_similarity:.3f}")
            st.write(f"Matched Question: {st.session_state.debug_matched_question}")
            st.write(f"Matched Answer: {st.session_state.debug_matched_answer}")

if __name__ == "__main__":
    main()