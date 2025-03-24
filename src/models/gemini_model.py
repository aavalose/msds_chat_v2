import google.generativeai as genai

class GeminiModel:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Google API key not found. Please configure it in your Streamlit secrets.")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")

    def _create_faculty_prompt(self, user_input, faculty_info):
        return f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
        
        A student has asked about faculty: "{user_input}"
        
        Please use the following faculty information to answer their question:
        
        ```
        {faculty_info}
        ```
        
        Please respond in a natural, conversational way while:
        1. Providing accurate information about the faculty members
        2. Being friendly and helpful
        3. Addressing their specific question directly
        4. Using clear and accessible language
        """

    def _create_general_prompt(self, user_input, general_info, retrieved_question, retrieved_answer, similarity_score):
        if retrieved_question and retrieved_answer and similarity_score >= 0.3:
            return f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program. 
            
            User question: "{user_input}"
            
            I found this similar question in our database (similarity: {similarity_score:.2f}):
            Question: "{retrieved_question}"
            Official answer: "{retrieved_answer}"
            
            Additional context:
            ```
            {general_info}
            ```
            
            Instructions:
            1. The matched question/answer pair has a similarity score of {similarity_score:.2f}
            2. If the similarity is high (>0.6), prioritize the official answer
            3. If the similarity is moderate (0.45-0.6), blend the official answer with general information
            4. Always maintain accuracy and be explicit about any uncertainty
            5. Use a friendly, conversational tone
            6. Address the specific aspects of the user's question
            
            Please provide a complete response that best answers the user's specific question."""
        else:
            return f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
            
            User question: "{user_input}"
            
            Please use this general information to help answer their question:
            ```
            {general_info}
            ```
            
            Instructions:
            1. If the answer can be found in the general information, provide a helpful and accurate response
            2. Only respond to questions related to the USF MSDS program
            3. If you don't have enough information:
               - Share what relevant information you do have
               - Acknowledge what you don't know
               - Suggest contacting the program office for those details
            4. Use a helpful and professional tone
            5. Be clear about what you know vs. what you're unsure about"""

    def generate_response(self, user_input, retrieved_question=None, 
                         retrieved_answer=None, similarity_score=None):
        try:
            general_info = open('general_info.txt', 'r').read()
            
            if any(term in user_input.lower() for term in ["faculty", "professor", "instructor"]):
                faculty_info = open('faculty.json', 'r').read()
                prompt = self._create_faculty_prompt(user_input, faculty_info)
            else:
                prompt = self._create_general_prompt(
                    user_input, general_info, retrieved_question, 
                    retrieved_answer, similarity_score
                )

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}" 