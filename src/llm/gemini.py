import os
import json
import streamlit as st
import google.generativeai as genai
from src.utils.preprocessing import preprocess_query
import re

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

def clean_response(text):
    """Clean and normalize the response text"""
    # First, fix common merging issues with specific phrases
    text = text.replace("whileinternationallyis", "while internationally is")
    text = text.replace("whileinternationally", "while internationally")
    
    # Replace Unicode characters with ASCII equivalents
    text = text.replace('′', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Fix spacing around specific words
    text = re.sub(r'while\s*internationally', 'while internationally', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)  # Fix number formatting
    
    # Add spaces around comparison words
    for word in ['while', 'and', 'in', 'is']:
        text = re.sub(f'([a-zA-Z0-9])({word})([a-zA-Z0-9])', f'\\1 {word} \\3', text, flags=re.IGNORECASE)
    
    # Ensure proper currency formatting
    text = re.sub(r'(\$\s*\d+)', lambda m: m.group(1).replace(' ', ''), text)
    
    # Clean up any double spaces
    text = ' '.join(text.split())
    
    return text

def get_gemini_response(user_input, retrieved_questions=None, retrieved_answers=None):
    try:
        # Get conversation history
        conversation_history = get_conversation_history()
        
        # Process the query to get categories
        processed_query, primary_category, all_categories = preprocess_query(user_input)
        
        # Load general information
        with open('data/general_info.txt', 'r') as f:
            general_info = f.read()
        
        # Load relevant category information from context.json
        try:
            with open('data/context.json', 'r') as f:
                context_data = json.load(f)
            
            # Get category-specific information but exclude qa_pairs to avoid redundancy
            category_info = {}
            for category in all_categories:
                if category in context_data:
                    # Create a copy without qa_pairs
                    category_info[category] = {k: v for k, v in context_data[category].items() if k != 'qa_pairs'}
        except Exception as e:
            category_info = {}
        
        # Format salary information specifically
        if "Career Outcomes" in category_info:
            salaries = category_info["Career Outcomes"].get("salaries", {})
            if salaries:
                # Create a formatted version of salary information
                salary_info = {
                    "median_base_salary_california": f"${salaries.get('median base salary in California', '').replace('$', '')}",
                    "median_base_salary_international": f"${salaries.get('median base salary internationally', '').replace('$', '')}",
                    "average_signing_bonus": f"${salaries.get('average signing bonus', '').replace('$', '')}"
                }
                category_info["Career Outcomes"]["salaries"] = salary_info
        
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
        
        When discussing salaries, you MUST use this EXACT template:
        "After graduating from the MSDS program, you can expect a competitive salary. Specifically, the median base salary in California is [amount], while internationally it is [amount]. The average signing bonus is [amount]."
        
        Always maintain proper spacing between words, especially:
        - Between numbers and words
        - Around the word "while"
        - Around the word "internationally"
        - Around the word "is"
        
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
        8. Ensure proper spacing between words and around punctuation
        9. Format currency values with the $ symbol and proper spacing
        10. When comparing values (e.g., "while internationally"), ensure proper word spacing
        
        Additional context:
        {general_info}
        
        Please provide your response:"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # Clean the response before returning
        cleaned_response = clean_response(response.text)
        return cleaned_response

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."
