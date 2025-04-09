import os
import json
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()

def get_mongo_client():
    """Get MongoDB client using connection string from Streamlit secrets or environment variables."""
    # Try to get connection string from Streamlit secrets first
    connection_string = st.secrets.get("MONGO_CONNECTION_STRING")
    
    # If not found in Streamlit secrets, try environment variables
    if not connection_string:
        connection_string = os.getenv("MONGODB_URI")
    
    if not connection_string:
        st.error("MongoDB connection string not found. Please set MONGO_CONNECTION_STRING in Streamlit secrets or MONGODB_URI in .env file.")
        return None
        
    try:
        client = MongoClient(
            connection_string,
            tls=True,
            tlsAllowInvalidCertificates=True,  # Allow invalid certificates for development
            serverSelectionTimeoutMS=30000,    # Increased from 5000 to 30000
            connectTimeoutMS=30000,            # Increased from 10000 to 30000
            socketTimeoutMS=45000,             # Added socket timeout
            retryWrites=True,
            maxPoolSize=50,
            waitQueueTimeoutMS=30000,          # Added wait queue timeout
            retryReads=True                    # Added retry reads
        )
        
        # Test the connection with a longer timeout
        client.admin.command('ping', serverSelectionTimeoutMS=30000)
        return client
    except ServerSelectionTimeoutError as e:
        st.error(f"Failed to connect to MongoDB: Server selection timeout. Please check your internet connection and MongoDB server status.")
        return None
    except ConnectionFailure as e:
        st.error(f"Failed to connect to MongoDB: Connection failure. Please verify your connection string and network settings.")
        return None
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

def init_mongodb():
    """Initialize MongoDB collections with data from files."""
    try:
        client = get_mongo_client()
        if client is None:
            st.error("Failed to initialize MongoDB: Could not establish connection")
            return None
            
        db = client.msds_chatbot
        
        # Initialize collections
        collections = {
            "courses": "data/courses.json",
            "faculty": "data/faculty.json",
            "qa_pairs": "data/labeled_qa.csv",
            "test_questions": "data/test_questions.csv"
        }
        
        for collection_name, file_path in collections.items():
            try:
                collection = db[collection_name]
                collection.delete_many({})  # Clear existing data
                
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            collection.insert_many(data)
                        else:
                            collection.insert_one(data)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    collection.insert_many(df.to_dict('records'))
            except Exception as e:
                st.error(f"Error initializing collection {collection_name}: {str(e)}")
        
        return db
    except Exception as e:
        st.error(f"Failed to initialize MongoDB: {str(e)}")
        return None

def save_conversation(session_id, user_message, bot_response, response_time, metrics=None, response_similarity=None):
    client = get_mongo_client()
    if client is None:
        return None
        
    try:
        db = client.msds_chatbot
        conversations_collection = db.conversations
        
        conversation = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "user_message": user_message,
            "bot_response": bot_response,
            "feedback": None,
            "similarity_score": st.session_state.debug_similarity,
            "matched_question": st.session_state.debug_matched_question,
            "response_time_seconds": response_time,
            "response_similarity": response_similarity
        }
        
        # Add metrics if available
        if metrics:
            conversation["metrics"] = metrics
            
        result = conversations_collection.insert_one(conversation)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error saving conversation to MongoDB: {str(e)}")
        return None

def update_feedback(conversation_id, feedback):
    client = get_mongo_client()
    if client is None:
        return
        
    try:
        db = client.msds_chatbot
        conversations_collection = db.conversations
        
        conversations_collection.update_one(
            {"_id": conversation_id},
            {"$set": {"feedback": feedback}}
        )
    except Exception as e:
        st.error(f"Error updating feedback: {str(e)}")
