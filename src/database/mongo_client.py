from pymongo import MongoClient
from datetime import datetime

class MongoDBClient:
    def __init__(self, connection_string):
        self.client = MongoClient(
            connection_string,
            tls=True,
            tlsAllowInvalidCertificates=False,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True,
            maxPoolSize=50
        )
        self.db = self.client.MSDSchatbot
        self.conversations = self.db.conversations

    def save_conversation(self, session_id, user_message, bot_response, 
                         similarity_score, matched_question, response_time):
        conversation = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "user_message": user_message,
            "bot_response": bot_response,
            "feedback": None,
            "similarity_score": similarity_score,
            "matched_question": matched_question,
            "response_time_seconds": response_time
        }
        result = self.conversations.insert_one(conversation)
        return str(result.inserted_id)

    def update_feedback(self, conversation_id, feedback):
        self.conversations.update_one(
            {"_id": conversation_id},
            {"$set": {"feedback": feedback}}
        ) 