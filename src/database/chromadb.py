import streamlit as st
import os
import json
import chromadb
from chromadb.utils import embedding_functions

@st.cache_resource
def init_chroma():
    try:
        # Create a persistent directory for ChromaDB
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize the client with persistence
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Use ChromaDB's default embedding function
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        return chroma_client, embedding_function
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        raise e

@st.cache_resource
def load_and_index_json_data(_chroma_client, _embedding_function, collection_name="msds_program_qa"):
    try:
        # Delete existing collection if it exists
        try:
            _chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            # Collection might not exist, which is fine
            pass
            
        # Create new collection
        qa_collection = _chroma_client.create_collection(
            name=collection_name,
            embedding_function=_embedding_function
        )

        # Load data from context.json file
        try:
            with open("data/context.json", "r") as f:
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
                                "Category": category,
                                "Answer": qa_pair["answer"],
                                "Type": "qa_pair"
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
                        "Category": category,
                        "Answer": summary,
                        "Type": "category_summary"
                    })
                    ids.append(f"{category.lower().replace(' ', '_')}_summary_{counter}")
                    counter += 1
            
            # Now add all the data to ChromaDB
            if documents:
                qa_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            else:
                st.warning("No documents were created from JSON data")

        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
            raise e

        return qa_collection
    except Exception as e:
        st.error(f"Error initializing QA collection from JSON: {str(e)}")
        raise e
