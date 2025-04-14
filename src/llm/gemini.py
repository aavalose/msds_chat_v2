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
    # First, remove any single-character spacing
    text = re.sub(r'\b(\w)\s+', r'\1', text)  # Remove spaces between single characters
    text = re.sub(r'\s+(\w)\b', r'\1', text)  # Remove spaces before single characters
    
    # Fix common salary formatting issues
    text = re.sub(r'(\d{3})\s*,\s*(\d{3})', r'\1,\2', text)  # Fix number formatting like "147 , 500"
    text = re.sub(r'([0-9]),([0-9])', r'\1,\2', text)  # Ensure proper comma formatting in numbers
    
    # Fix specific word merging issues
    text = re.sub(r'while\s*internationally\s*it\s*is', 'while internationally it is', text, flags=re.IGNORECASE)
    text = re.sub(r'sign\s*in\s*g', 'signing', text)  # Fix "sign in g" -> "signing"
    text = re.sub(r'graduat\s*in\s*g', 'graduating', text)  # Fix "graduat in g" -> "graduating"
    
    # Ensure proper spacing around currency
    text = re.sub(r'\$\s+', '$', text)  # Remove space after $
    
    # Clean up any double spaces and normalize spacing
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

        # Set up the generation config to help prevent formatting issues
        generation_config = {
            "temperature": 0.1,  # Lower temperature for more consistent formatting
            "top_p": 0.8,
            "top_k": 40,
            "candidate_count": 1,
        }

        # Create the safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DEROGATORY",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_TOXICITY",
                "threshold": "BLOCK_NONE",
            },
        ]

        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate response with specific configuration
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Get the text and clean it
        response_text = response.text
        
        # Additional cleaning for salary-specific responses
        if "salary" in user_input.lower() or "Career Outcomes" in str(category_info):
            # Extract salary information from category_info
            if "Career Outcomes" in category_info and "salaries" in category_info["Career Outcomes"]:
                salaries = category_info["Career Outcomes"]["salaries"]
                # Force the exact format we want
                response_text = (
                    f"After graduating from the MSDS program, you can expect a competitive salary. "
                    f"Specifically, the median base salary in California is {salaries['median_base_salary_california']}, "
                    f"while internationally it is {salaries['median_base_salary_international']}. "
                    f"The average signing bonus is {salaries['average_signing_bonus']}."
                )
        
        # Clean the response
        cleaned_response = clean_response(response_text)
        
        # Final verification of formatting
        if "salary" in cleaned_response.lower():
            # Ensure proper spacing in final output
            cleaned_response = re.sub(r'(\d),(\d)', r'\1,\2', cleaned_response)
            cleaned_response = re.sub(r'(\d)(while)', r'\1 \2', cleaned_response)
            cleaned_response = re.sub(r'(is)(\d)', r'\1 \2', cleaned_response)
        
        return cleaned_response

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."
