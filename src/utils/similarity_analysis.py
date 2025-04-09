import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text for better comparison"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def enhanced_cosine_similarity(bot_resp, expert_resp):
    """Calculate enhanced cosine similarity focusing on key terms"""
    custom_stopwords = [
        'thanks', 'hello', 'hi', 'question', 'great', 'thank', 'you', 'hope', 'helpful', 'please',
        'actually', 'just', 'would', 'could', 'may', 'might', 'glad', 'happy', 'can'
    ]
    
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), 
                           token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
    
    # Adding both texts to ensure the same vocabulary
    tfidf_matrix = tfidf.fit_transform([bot_resp, expert_resp])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def keyword_inclusion_score(bot_resp, expert_resp):
    """Calculate how many expert keywords are present in the bot response"""
    expert_tokens = set(expert_resp.lower().split())
    bot_tokens = set(bot_resp.lower().split())
    
    # Filter out common stopwords
    stopwords = set([
        'the', 'is', 'in', 'it', 'and', 'to', 'a', 'of', 'for', 'are', 'be',
        'this', 'that', 'with', 'on', 'at', 'by', 'from', 'an', 'as', 'or'
    ])
    
    expert_keywords = expert_tokens - stopwords
    
    # Count how many expert keywords are in the bot response
    matches = sum(1 for word in expert_keywords if word in bot_tokens)
    
    # Return percentage of expert keywords included
    if len(expert_keywords) == 0:
        return 1.0  # Perfect match if no keywords to match
    
    return matches / len(expert_keywords)

def semantic_similarity(bot_resp, expert_resp):
    """Calculate semantic similarity using Sentence Transformers"""
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate embeddings
    bot_embedding = model.encode(bot_resp, convert_to_tensor=True)
    expert_embedding = model.encode(expert_resp, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(bot_embedding.unsqueeze(0), 
                                                   expert_embedding.unsqueeze(0))
    
    return cos_sim.item()

def rouge_score(bot_resp, expert_resp):
    """Calculate ROUGE-L score"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    try:
        scores = scorer.score(bot_resp, expert_resp)
        # Using ROUGE-L which focuses on longest common subsequence
        return scores['rougeL'].fmeasure
    except:
        # Handle cases where ROUGE calculation might fail
        return 0.0

def bleu_score(bot_resp, expert_resp):
    """Calculate BLEU score"""
    # Tokenize sentences
    bot_tokens = word_tokenize(bot_resp.lower())
    expert_tokens = word_tokenize(expert_resp.lower())
    
    # Calculate BLEU score
    try:
        return sentence_bleu([expert_tokens], bot_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    except:
        return 0.0

def fact_matching_score(bot_resp, expert_resp):
    """Calculate fact matching score based on numbers and key phrases"""
    # Extract numbers
    numbers_pattern = r'\b\d+\.?\d*\b'
    bot_numbers = set(re.findall(numbers_pattern, bot_resp))
    expert_numbers = set(re.findall(numbers_pattern, expert_resp))
    
    # Check for job titles or key phrases
    key_phrases = [
        'data scientist', 'machine learning', 'gpa', 'average', 'prerequisite',
        'teaching assistant', 'research assistant', 'double dons', 'scholarship', 
        'march 1st', 'deadline', 'admitted', 'international', 'tuition', '20%'
    ]
    
    bot_phrases = set([phrase for phrase in key_phrases if phrase.lower() in bot_resp.lower()])
    expert_phrases = set([phrase for phrase in key_phrases if phrase.lower() in expert_resp.lower()])
    
    # Combine and calculate score
    total_expert_items = len(expert_numbers) + len(expert_phrases)
    matched_items = len(bot_numbers.intersection(expert_numbers)) + len(bot_phrases.intersection(expert_phrases))
    
    if total_expert_items == 0:
        return 1.0  # Perfect match if no items to match
        
    return matched_items / total_expert_items

def calculate_all_metrics(bot_response, expert_response):
    """Calculate all similarity metrics for a response pair"""
    # Preprocess responses
    bot_resp_processed = preprocess_text(bot_response)
    expert_resp_processed = preprocess_text(expert_response)
    
    # Calculate all metrics
    metrics = {
        'enhanced_cosine_similarity': enhanced_cosine_similarity(bot_resp_processed, expert_resp_processed),
        'keyword_inclusion': keyword_inclusion_score(bot_resp_processed, expert_resp_processed),
        'semantic_similarity': semantic_similarity(bot_response, expert_response),
        'rouge_l_score': rouge_score(bot_resp_processed, expert_resp_processed),
        'bleu_score': bleu_score(bot_resp_processed, expert_resp_processed),
        'fact_matching': fact_matching_score(bot_response, expert_response),
        'timestamp': datetime.now()
    }
    
    # Calculate average score (excluding BLEU score)
    metrics['average_score'] = np.mean([
        metrics['enhanced_cosine_similarity'],
        metrics['keyword_inclusion'],
        metrics['semantic_similarity'],
        metrics['rouge_l_score'],
        metrics['fact_matching']
    ])
    
    return metrics

def analyze_conversation(bot_response, question, expert_response):
    """Analyze a conversation and return metrics"""
    metrics = calculate_all_metrics(bot_response, expert_response)
    
    # Add question and responses to metrics
    metrics.update({
        'question': question,
        'bot_response': bot_response,
        'expert_response': expert_response
    })
    
    return metrics 