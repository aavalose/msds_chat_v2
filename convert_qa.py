import csv
import re
import os

def convert_qa_to_csv(input_file, output_file):
    # First, read existing questions from CSV if it exists
    existing_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_questions = {row['Question'].strip() for row in reader}
    
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split the content into Q&A pairs
    questions = re.findall(r'Q:\s*(.*?)\nA:', content, re.DOTALL)
    answers = re.findall(r'A:\s*(.*?)(?=(?:\nQ:|$))', content, re.DOTALL)
    
    # Combine questions and answers
    qa_pairs = list(zip(questions, answers))
    
    # Clean up the pairs and filter out existing questions
    new_pairs = []
    for q, a in qa_pairs:
        cleaned_q = q.strip()
        if cleaned_q not in existing_questions:
            new_pairs.append((cleaned_q, a.strip()))
    
    # If file doesn't exist, create it with headers
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'Answer'])
    
    # Append new pairs to CSV if there are any
    if new_pairs:
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_pairs)
        print(f"Added {len(new_pairs)} new Q&A pairs to the CSV file.")
    else:
        print("No new questions to add.")

if __name__ == "__main__":
    convert_qa_to_csv('february_text.txt', 'Questions_and_Answers.csv') 