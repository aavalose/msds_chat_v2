import os
import json
import streamlit as st
import google.generativeai as genai
from src.utils.preprocessing import preprocess_query

# Configure Gemini model
@st.cache_resource
def load_gemini_model():
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

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

def get_gemini_response(user_input, retrieved_questions=None, retrieved_answers=None):
    try:
        conversation_history = get_conversation_history()
        processed_query, primary_category, all_categories = preprocess_query(user_input)
        
        # Load general information
        with open('data/general_info.txt', 'r') as f:
            general_info = f.read()
        
        # Load relevant category information
        try:
            with open('data/context.json', 'r') as f:
                context_data = json.load(f)
            category_info = {
                category: {k: v for k, v in context_data[category].items() if k != 'qa_pairs'}
                for category in all_categories
                if category in context_data
            }
        except Exception:
            category_info = {}
        
        # Format salary information if needed
        if "Career Outcomes" in category_info:
            salaries = category_info["Career Outcomes"].get("salaries", {})
            if salaries:
                salary_info = {
                    "median_base_salary_california": f"${salaries.get('median base salary in California', '').replace('$', '')}",
                    "median_base_salary_international": f"${salaries.get('median base salary internationally', '').replace('$', '')}",
                    "average_signing_bonus": f"${salaries.get('average signing bonus', '').replace('$', '')}"
                }
                category_info["Career Outcomes"]["salaries"] = salary_info
        
        # Format relevant QA pairs
        relevant_qa_pairs = ""
        if retrieved_questions and retrieved_answers and st.session_state.debug_similarity >= 0.3:
            if not isinstance(retrieved_questions, list):
                retrieved_questions = [retrieved_questions]
                retrieved_answers = [retrieved_answers]
            relevant_qa_pairs = "\n\nRelevant information from our database:\n"
            for q, a in zip(retrieved_questions, retrieved_answers):
                relevant_qa_pairs += f"Q: {q}\nA: {a}\n"

        prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.

Current conversation:
{conversation_history}

User question: "{user_input}"
Category: {primary_category}
Related Categories: {', '.join(all_categories[1:]) if len(all_categories) > 1 else 'None'}

Program Information:
{general_info}

{json.dumps(category_info, indent=2) if category_info else ""}
{relevant_qa_pairs}

Answer the user's question based on the provided information. Be friendly, concise, and accurate.
If you don't know the answer, say so politely and suggest contacting the program directly.
Do not make up information that is not in the provided context.
"""

        # Get response from Gemini model
        model = load_gemini_model()
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or contact the program directly for assistance."