import streamlit as st
import os
import json
import chromadb
from chromadb.utils import embedding_functions
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def init_chroma():
    try:
        # Define the ChromaDB directory path
        chroma_dir = os.path.join(os.getcwd(), "chroma_db")
        logger.info(f"Initializing ChromaDB in directory: {chroma_dir}")
        
        # Clear existing ChromaDB directory to start fresh
        if os.path.exists(chroma_dir):
            logger.info("Clearing existing ChromaDB directory")
            shutil.rmtree(chroma_dir)
        
        # Create a persistent directory for ChromaDB
        os.makedirs(chroma_dir, exist_ok=True)
        logger.info("Created ChromaDB directory")
        
        # Initialize the client with persistence only
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_dir)
            logger.info("Successfully initialized ChromaDB client")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
        
        # Use ChromaDB's default embedding function
        try:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Successfully initialized embedding function")
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {str(e)}")
            raise
        
        return chroma_client, embedding_function
    except Exception as e:
        logger.error(f"Error in ChromaDB initialization: {str(e)}")
        st.error(f"Error initializing ChromaDB: {str(e)}")
        raise

@st.cache_resource
def load_and_index_json_data(_chroma_client, _embedding_function, collection_name="msds_program_qa"):
    try:
        logger.info(f"Loading and indexing data for collection: {collection_name}")
        
        # Delete existing collection if it exists
        try:
            _chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            logger.info(f"No existing collection to delete: {str(e)}")
            
        # Create new collection with explicit metadata schema
        try:
            qa_collection = _chroma_client.create_collection(
                name=collection_name,
                embedding_function=_embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

        # Load data from context.json file
        try:
            context_path = os.path.join(os.getcwd(), "data", "context.json")
            logger.info(f"Loading context data from: {context_path}")
            
            with open(context_path, "r") as f:
                context_data = json.load(f)
            
            # Generate documents for ChromaDB from JSON data
            documents = []
            metadatas = []
            ids = []
            counter = 0
            
            # Process each category in context.json
            for category, data in context_data.items():
                # If the category has QA pairs, add them to the collection
                if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
                    for qa_pair in data["qa_pairs"]:
                        if "question" in qa_pair and "answer" in qa_pair:
                            documents.append(qa_pair["question"])
                            metadatas.append({
                                "category": category,
                                "answer": qa_pair["answer"],
                                "type": "qa_pair"
                            })
                            ids.append(f"{category.lower().replace(' ', '_')}_{counter}")
                            counter += 1
                
                # Also create broader category-based questions
                category_questions = [
                    f"Tell me about {category}",
                    f"What is the {category} like?",
                    f"Information about {category}"
                ]
                
                # Create a summary of the category data
                summary = json.dumps(data, ensure_ascii=False)
                if len(summary) > 1000:  # If too long, create a shorter version
                    # Remove qa_pairs for the summary to keep it focused on structured data
                    summary_data = {k: v for k, v in data.items() if k != 'qa_pairs'}
                    summary = json.dumps(summary_data, ensure_ascii=False)
                
                for question in category_questions:
                    documents.append(question)
                    metadatas.append({
                        "category": category,
                        "answer": summary,
                        "type": "category_summary"
                    })
                    ids.append(f"{category.lower().replace(' ', '_')}_summary_{counter}")
                    counter += 1
            
            # Now add all the data to ChromaDB
            if documents:
                try:
                    qa_collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.info(f"Successfully added {len(documents)} documents to collection")
                except Exception as e:
                    logger.error(f"Failed to add documents to collection: {str(e)}")
                    raise
            else:
                logger.warning("No documents were created from JSON data")
                st.warning("No documents were created from JSON data")

        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            st.error(f"Error loading JSON data: {str(e)}")
            raise

        return qa_collection
    except Exception as e:
        logger.error(f"Error in load_and_index_json_data: {str(e)}")
        st.error(f"Error initializing QA collection from JSON: {str(e)}")
        raise
