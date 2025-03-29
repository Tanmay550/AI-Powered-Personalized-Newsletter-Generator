import streamlit as st
import feedparser
import torch
from transformers import pipeline
import datetime
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Check CUDA availability, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if device == "cuda" else -1

# Load AI models (with caching to improve speed)
@st.cache_resource()
def load_models():
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id, framework="pt")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_id, framework="pt")
    except Exception:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1, framework="pt")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1, framework="pt")
    return classifier, summarizer

classifier, summarizer = load_models()

# RSS Feeds Dictionary
RSS_FEEDS = {

    "BBC_World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "NYTimes_World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
    "TechCrunch": "http://feeds.feedburner.com/TechCrunch/",
    "Wired": "https://www.wired.com/feed/rss",
    "MIT_Technology_Review": "https://www.technologyreview.com/feed/",
    "Bloomberg": "https://www.bloomberg.com/feed/",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Financial_Times": "https://www.ft.com/?format=rss",
    "ESPN": "https://www.espn.com/espn/rss/news",
    "BBC_Sport": "http://feeds.bbci.co.uk/sport/rss.xml?edition=uk",
    "Sky_Sports": "https://www.skysports.com/rss/12040",
    "Variety": "https://variety.com/feed/",
    "Hollywood_Reporter": "https://www.hollywoodreporter.com/t/hollywood/feed/",
    "Billboard": "https://www.billboard.com/feed/",
    "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "Science_Daily": "https://www.sciencedaily.com/rss/rss/top/science.xml",
    "Ars_Technica_Science": "https://arstechnica.com/science/",
}

# AI-generated categories
DYNAMIC_CATEGORIES = ["Politics", "Technology", "Finance", "Sports", "Entertainment", "Science", "Health", "World News"]

# Fetch RSS Articles (with caching)
@st.cache_data()
def fetch_rss_articles(url):
    feed = feedparser.parse(url)
    return [{
        "title": entry.get("title", "No Title"),
        "summary": entry.get("summary", "No Summary"),
        "link": entry.get("link", "No Link"),
        "source": url
    } for entry in feed.entries[:5]] if feed.entries else []

# Classify Articles
@st.cache_data()
def classify_article(text):
    text = text[:512]
    try:
        result = classifier(text, DYNAMIC_CATEGORIES, multi_label=False)
        return result["labels"][0]
    except Exception:
        return "World News"

# Summarize Text
@st.cache_data()
def summarize_text(text):
    if len(text.split()) > 50:
        try:
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
            return summary[0]["summary_text"]
        except Exception:
            return text
    return text

# Generate Newsletter
@st.cache_data()
def generate_newsletter(articles_by_category):
    # Ensure all articles have summaries
    all_summaries = " ".join([summarize_text(article["summary"]) for articles in articles_by_category.values() for article in articles])

    # Generate an overall summary of all articles
    general_summary = "No overall summary available."
    if len(all_summaries.split()) > 50:
        try:
            general_summary = summarizer(all_summaries, max_length=250, min_length=100, do_sample=False)[0]["summary_text"]
        except Exception:
            pass

    # Extract top 3 trending/highlighted articles (based on first appearance)
    top_articles = []
    for category, articles in articles_by_category.items():
        for article in articles:
            top_articles.append(f"- **{article['title']}** ({category})")
        if len(top_articles) >= 3:
            break  # Limit to 3 top articles

    trending_highlights = "\n".join(top_articles[:3]) if top_articles else "No trending highlights available."

    # Build newsletter
    newsletter = f"# 📰 AI-Powered News Digest ({datetime.date.today()})\n\n"
    newsletter += "## 🌟 Trending Highlights\n"
    newsletter += f"{trending_highlights}\n\n"
    newsletter += "---\n\n"
    
    # newsletter += "## 📌 General Summary\n"
    # newsletter += f"{general_summary}\n\n"
    # newsletter += "---\n\n"

    newsletter += "## 🗂 Categorized News\n"
    for category, articles in articles_by_category.items():
        newsletter += f"\n## {category}\n"
        for article in articles:
            newsletter += f"### {article['title']}\n"
            newsletter += f"**Source:** {article['source']}\n\n"
            newsletter += f"**Summary:** {article['summary']}\n\n"
            newsletter += f"[Read more]({article['link']})\n\n"

    return newsletter

# Streamlit UI
st.title("📢 AI-Powered Dynamic News Categorization")
st.sidebar.header("User Preferences")
selected_categories = st.sidebar.multiselect("Choose categories", DYNAMIC_CATEGORIES)

if st.sidebar.button("Fetch News"):
    if not selected_categories:
        st.warning("⚠️ Please select at least one category.")
    else:
        st.subheader("📰 AI-Categorized News Feed")
        articles_by_category = {category: [] for category in selected_categories}
        
        progress_bar = st.progress(0)
        total_sources = len(RSS_FEEDS)
        
        for i, (source, url) in enumerate(RSS_FEEDS.items()):
            articles = fetch_rss_articles(url)
            for article in articles:
                text = f"{article['title']} {article['summary']}"
                category = classify_article(text)
                if category in selected_categories:
                    article["summary"] = summarize_text(article["summary"])
                    articles_by_category[category].append(article)
                    st.markdown(f"## 📌 {category} News")
                    st.markdown(f"### 🔍 {source}")
                    st.markdown(f"**📰 {article['title']}**")
                    st.write(f"📄 **Summary:** {article['summary']}")
                    st.markdown(f"[Read more...]({article['link']})")
                    st.write("---")
            progress_bar.progress((i + 1) / total_sources)
        
        progress_bar.empty()
        
        # Generate and download newsletter
        newsletter_content = generate_newsletter(articles_by_category)
        st.download_button(label="📩 Download Newsletter", data=newsletter_content.encode("utf-8"), file_name=f"AI_Newsletter_{datetime.date.today()}.md", mime="text/markdown")
