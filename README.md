# AI Powered Personalized Newsletter Generator

## Overview
This project is an NLP-based Newsletter Generator that creates custom newsletters based on user preferences. The system dynamically categorizes and summarizes articles fetched from RSS feeds.

## Features:
- 📰 Personalized Content: Generates a custom newsletter based on user-selected categories.
- 🔄 Real-time Updates: Uses RSS feeds to gather up-to-date articles from various domains (Technology, Science, Sports, etc.).
- 🤖 AI-Powered Categorization: Implements the bart-large-mnli model to classify articles dynamically.
- ✍️ Concise Summaries: Uses the bart-large-cnn model to generate summaries for each article.
- 📌 Well-structured Newsletter:
    1. A concise summary highlighting the most important/trending articles.
    2. A well-organized layout with sections based on topics.
    3. Summaries for selected articles with key points.
    4. Hyperlinked references directing users to the full articles.

## Installation & Setup

Ensure you have the following installed:
1. Python 3.8+
2. Required dependencies:
`pip install streamlit transformers feedparser beautifulsoup4 torch`


## Running the Application
1. Clone the Repository:
`git clone <repo-url>`
`cd <repo-folder>`

2. Launch the Streamlit App:
`streamlit run newsletter_generator.py`

## AI Models Used
1. BART-Large-MNLI: Categorizes articles dynamically.
2. BART-Large-CNN: Summarizes article content.




