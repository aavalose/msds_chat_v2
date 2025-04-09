import os
import json
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson.objectid import ObjectId

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
            
        db = client.MSDSchatbot
        conversations_collection = db.conversations
        
        # Test the collection by inserting and removing a test document
        try:
            test_result = conversations_collection.insert_one({"test": "connection"})
            conversations_collection.delete_one({"_id": test_result.inserted_id})
        except Exception as e:
            st.error(f"Failed to test MongoDB collection access: {str(e)}")
            return None
        
        return conversations_collection
    except Exception as e:
        st.error(f"Failed to initialize MongoDB: {str(e)}")
        return None

def save_conversation(session_id, user_message, bot_response, response_time, metrics=None, response_similarity=None):
    client = get_mongo_client()
    if client is None:
        return None
        
    try:
        db = client.MSDSchatbot
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
        db = client.MSDSchatbot
        conversations_collection = db.conversations
        
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(conversation_id)
        except Exception as e:
            st.error(f"Invalid conversation ID format: {str(e)}")
            return
        
        result = conversations_collection.update_one(
            {"_id": obj_id},
            {"$set": {"feedback": feedback}}
        )
        
        if result.modified_count > 0:
            st.success("Successfully updated feedback")
        else:
            st.warning("No conversation found with the given ID")
            
    except Exception as e:
        st.error(f"Error updating feedback: {str(e)}")

def get_conversation(conversation_id):
    """Utility function to retrieve a conversation by ID for debugging."""
    client = get_mongo_client()
    if client is None:
        return None
        
    try:
        db = client.MSDSchatbot
        conversations_collection = db.conversations
        
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(conversation_id)
        except Exception as e:
            st.error(f"Invalid conversation ID format: {str(e)}")
            return None
        
        conversation = conversations_collection.find_one({"_id": obj_id})
        return conversation
    except Exception as e:
        st.error(f"Error retrieving conversation: {str(e)}")
        return None

def save_metrics(metrics):
    """Save similarity analysis metrics to MongoDB"""
    client = get_mongo_client()
    if client is None:
        return None
        
    try:
        db = client.MSDSchatbot
        metrics_collection = db.metrics
        
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now()
            
        result = metrics_collection.insert_one(metrics)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error saving metrics to MongoDB: {str(e)}")
        return None
