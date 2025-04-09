# USF MSDS Program Chatbot

A conversational AI assistant for the University of San Francisco's Master of Science in Data Science (MSDS) program. This chatbot helps prospective and current students get accurate information about the program using advanced natural language processing and retrieval-augmented generation.

## Features

- **Instant Answers**: Get immediate responses to questions about admissions, curriculum, faculty, and more
- **Smart Retrieval**: Uses ChromaDB for vector storage and semantic search
- **Conversation Memory**: Remembers previous questions in the current session
- **Feedback System**: Users can provide feedback on responses
- **Analytics**: Stores conversations and feedback in MongoDB for analysis

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── src/                    # Source code directory
│   ├── database/          # Database-related modules
│   │   ├── mongodb.py     # MongoDB operations
│   │   └── chromadb.py    # ChromaDB operations
│   ├── llm/               # LLM-related modules
│   │   └── gemini.py      # Gemini model operations
│   ├── retrieval/         # Retrieval-related modules
│   │   └── qa_retrieval.py # Question-answer retrieval
│   └── utils/             # Utility modules
│       ├── preprocessing.py # Text preprocessing
│       └── similarity.py   # Text similarity calculations
├── context.json           # Program information and QA pairs
├── general_info.txt       # General program information
└── images/                # Images for the UI
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MSDS-chatbot.git
cd MSDS-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.streamlit/secrets.toml` file with:
```toml
GOOGLE_API_KEY = "your_google_api_key"
MONGO_CONNECTION_STRING = "your_mongodb_connection_string"
```

5. Run the application:
```bash
streamlit run app.py
```

## Data Files

- `context.json`: Contains structured information about the MSDS program, including QA pairs and category information
- `general_info.txt`: Contains general information about the program
- `images/`: Contains images used in the UI (USF logo, developer headshots)

## Technologies Used

- **Streamlit**: Web application framework
- **Google Gemini AI**: Natural language understanding and generation
- **ChromaDB**: Vector storage and semantic search
- **MongoDB**: Conversation logging and analytics
- **Scikit-learn**: Text similarity calculations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed by Sehej Singh and Arturo Avalos
- Special thanks to the USF MSDS program faculty and staff 