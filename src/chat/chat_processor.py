from datetime import datetime

class ChatProcessor:
    def __init__(self, gemini_model, similarity_client, mongo_client):
        self.gemini_model = gemini_model
        self.similarity_client = similarity_client
        self.mongo_client = mongo_client

    def process_message(self, user_input, session_id):
        start_time = datetime.now()
        
        # Find similar question
        matched_question, matched_answer, similarity = \
            self.similarity_client.find_similar_question(user_input)

        # Generate response
        bot_response = self.gemini_model.generate_response(
            user_input, matched_question, matched_answer, similarity
        )

        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()

        # Save conversation
        conversation_id = self.mongo_client.save_conversation(
            session_id, user_input, bot_response, 
            similarity, matched_question, response_time
        )

        return bot_response, conversation_id, {
            'similarity': similarity,
            'matched_question': matched_question,
            'matched_answer': matched_answer
        }

    def update_feedback(self, conversation_id, feedback):
        """Update feedback for a conversation"""
        self.mongo_client.update_feedback(conversation_id, feedback) 