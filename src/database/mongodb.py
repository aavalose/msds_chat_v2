import os
import json
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from datetime import datetime
# from dotenv import load_dotenv # No longer needed
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson.objectid import ObjectId

# load_dotenv() # No longer needed

def get_mongo_client():
    print("DEBUG_LOG: Entered get_mongo_client() - using st.secrets exclusively.")
    st.write("DEBUG_UI: Entered get_mongo_client() - using st.secrets exclusively.")
    connection_string = st.secrets.get("MONGO_CONNECTION_STRING")
    
    print(f"DEBUG_LOG: st.secrets.get('MONGO_CONNECTION_STRING') returned: '{connection_string}'")
    st.write(f"DEBUG_UI: st.secrets.get('MONGO_CONNECTION_STRING') returned: '{connection_string}'")

    if not connection_string:
        print("DEBUG_LOG: MONGO_CONNECTION_STRING not found in st.secrets.")
        st.write("DEBUG_UI: MONGO_CONNECTION_STRING not found in st.secrets.")
        st.error("MongoDB connection string not found in st.secrets. Please ensure MONGO_CONNECTION_STRING is in secrets.toml.")
        return None
    
    print(f"DEBUG_LOG: Attempting to connect with connection_string from st.secrets: '{connection_string}'")
    st.write(f"DEBUG_UI: Attempting to connect with connection_string from st.secrets: '{connection_string}'")
    try:
        client = MongoClient(
            connection_string,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=45000,
            retryWrites=True,
            maxPoolSize=50,
            waitQueueTimeoutMS=30000,
            retryReads=True
        )
        client.admin.command('ping', serverSelectionTimeoutMS=30000)
        print("DEBUG_LOG: MongoDB client ping successful.")
        st.write("DEBUG_UI: MongoDB client ping successful.")
        return client
    except Exception as e:
        print(f"DEBUG_LOG: Failed to connect to MongoDB: {str(e)}")
        st.write(f"DEBUG_UI: Failed to connect to MongoDB: {str(e)}")
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

def init_mongodb():
    print("DEBUG_LOG: Entered init_mongodb() - using st.secrets exclusively.")
    st.write("DEBUG_UI: Entered init_mongodb() - using st.secrets exclusively.")
    try:
        client = get_mongo_client()
        if client is None:
            # Error already shown by get_mongo_client or this function's st.write
            return None
        db = client.MSDSchatbot
        conversations_collection = db.conversations
        try:
            test_result = conversations_collection.insert_one({"test": "connection"})
            conversations_collection.delete_one({"_id": test_result.inserted_id})
            print("DEBUG_LOG: MongoDB collection access test successful.")
            st.write("DEBUG_UI: MongoDB collection access test successful.")
        except Exception as e:
            print(f"DEBUG_LOG: Failed to test MongoDB collection access: {str(e)}")
            st.write(f"DEBUG_UI: Failed to test MongoDB collection access: {str(e)}")
            st.error(f"Failed to test MongoDB collection access: {str(e)}")
            return None
        return conversations_collection
    except Exception as e:
        print(f"DEBUG_LOG: Failed to initialize MongoDB in init_mongodb: {str(e)}")
        st.write(f"DEBUG_UI: Failed to initialize MongoDB in init_mongodb: {str(e)}")
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
