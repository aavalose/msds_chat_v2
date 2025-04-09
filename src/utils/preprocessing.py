import streamlit as st
import google.generativeai as genai

def preprocess_query(query):
    processed_query = query.lower().strip()
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Analyze this question and return up to THREE most relevant categories from the following list, ordered by relevance:
        - Application Process: Questions about how to apply, deadlines, interviews, and application components
        - Admission Requirements: Questions about prerequisites, qualifications, and requirements
        - Financial Aid & Scholarships: Questions about funding, scholarships, and financial assistance
        - International Students: Questions specific to international student needs
        - Enrollment Process: Questions about post-acceptance procedures
        - Program Structure: Questions about program duration, format, and class sizes
        - Program Overview: Questions about general program information and features
        - Tuition & Costs: Questions about program costs, fees, and expenses
        - Program Preparation: Questions about preparing for the program
        - Faculty & Research: Questions about professors and research opportunities
        - Student Employment: Questions about work opportunities during the program
        - Student Services: Questions about health insurance and student support
        - Curriculum: Questions about courses and academic content
        - Practicum Experience: Questions about industry projects and partnerships
        - Career Outcomes: Questions about job placement, salaries, and career paths
        - Admission Statistics: Questions about typical GPAs, backgrounds, and work experience
        - Other: Questions that don't clearly fit into any of the above categories
        
        Examples:
        Question: "What GRE score do I need as an international student?" -> ["Application Process", "International Students"]
        Question: "How much is tuition and what scholarships are available?" -> ["Tuition & Costs", "Financial Aid & Scholarships"]
        Question: "Can I work while taking classes in the program?" -> ["Student Employment", "Program Structure"]
        Question: "Where is the nearest coffee shop?" -> ["Other"]
        
        Your question: "{query}"
        
        Return only the category names in a comma-separated list, nothing else."""
        
        response = model.generate_content(prompt)
        categories = [cat.strip() for cat in response.text.split(',')]
        primary_category = categories[0] if categories else "Other"
            
        return processed_query, primary_category, categories
    except Exception as e:
        return processed_query, "Other", ["Other"]
