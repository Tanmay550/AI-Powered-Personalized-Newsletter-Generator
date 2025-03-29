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

# Load AI models (no caching, always loads fresh)
def load_models():
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id, framework="pt")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_id, framework="pt")
    except Exception:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1, framework="pt")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1, framework="pt")
    return classifier, summarizer

classifier, summarizer = load_models()

# RSS Feeds Dictionary Categorized
RSS_FEEDS = {
    "General News": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "http://feeds.reuters.com/reuters/topNews"
    ],
    "Technology": [
        "http://feeds.feedburner.com/TechCrunch/",
        "https://www.wired.com/feed/rss",
        "https://www.technologyreview.com/feed/"
    ],
    "Finance": [
        "https://www.bloomberg.com/feed/",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.ft.com/business-education?format=rss"
    ],
    "Sports": [
        "https://rss.app/rss-feed?keyword=ESPN&region=US&lang=en",
        "https://rss.app/feeds/tWwnqzFBv5TWuhb2.xml",
        "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/mlb",
        "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/nba",
        "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/soccer,soccer/epl/league/1,soccer/mls/league/5,soccer/ucl/league/7,soccer/europa/league/8,soccer/wc/league/12,soccer/euro/league/13,soccer/wwc/league/14,soccer/nwsl/league/20,soccer/cwc/league/26,soccer/gold_cup/league/32,soccer/unl/league/67"
    ],
    "Entertainment": [
        "https://variety.com/feed/",
        "https://www.hollywoodreporter.com/t/hollywood/feed/",
        "https://www.billboard.com/feed/"
    ],
    "Science": [
        "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "https://www.sciencedaily.com/rss/all.xml",
        "http://feeds.arstechnica.com/arstechnica/science"
    ]
}

# Fetch RSS Articles
def fetch_rss_articles(url):
    feed = feedparser.parse(url)
    return [{
        "title": entry.get("title", "No Title"),
        "summary": entry.get("summary", "No Summary"),
        "link": entry.get("link", "No Link"),
        "source": url
    } for entry in feed.entries[:5]] if feed.entries else []

# Classify Articles
def classify_article(text, user_preferences, source):
    text = text[:512]  # Limit text length for better classification
    try:
        result = classifier(text, user_preferences, multi_label=True)
        category = result["labels"][0]
        if category not in user_preferences:
            return None
    except Exception:
        return None
    return category

# Summarize Text
def summarize_text(text):
    if len(text.split()) > 50:
        try:
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
            return summary[0]["summary_text"]
        except Exception:
            return text
    return text

# Generate Newsletter
def generate_newsletter(articles_by_category):
    # Ensure all articles have summaries
    all_summaries = " ".join([summarize_text(article["summary"]) for articles in articles_by_category.values() for article in articles])

    # Generate an overall summary of all articles
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
    
    newsletter += "## 📌 General Summary\n"
    newsletter += f"{general_summary}\n\n"
    newsletter += "---\n\n"

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

# User selects categories
selected_categories = st.sidebar.multiselect("Select Categories", list(RSS_FEEDS.keys()), default=["General News", "Technology"])

if st.sidebar.button("Fetch News"):
    if not selected_categories:
        st.warning("⚠️ Please select at least one category.")
    else:
        st.subheader("📰 AI-Categorized News Feed")
        articles_by_category = {category: [] for category in selected_categories}
        
        progress_bar = st.progress(0)
        total_sources = sum(len(RSS_FEEDS[cat]) for cat in selected_categories)
        processed_sources = 0
        
        for category in selected_categories:
            for url in RSS_FEEDS[category]:
                articles = fetch_rss_articles(url)
                for article in articles:
                    text = f"{article['title']} {article['summary']}"
                    classified_category = classify_article(text, selected_categories, category)
                    if classified_category:
                        article["summary"] = summarize_text(article["summary"])
                        articles_by_category.setdefault(classified_category, []).append(article)
                processed_sources += 1
                progress_bar.progress(processed_sources / total_sources)
        
        progress_bar.empty()
        
       
        for category, articles in articles_by_category.items():
            if articles:
                st.markdown(f"## 📌 {category} News")
                for article in articles:
                    st.markdown(f"### 🔍 {article['title']}")
                    st.write(f"📄 **Summary:** {article['summary']}")
                    st.markdown(f"[Read more...]({article['link']})")
                    st.write("---")
        newsletter_content = generate_newsletter(articles_by_category)
        st.download_button(label="📩 Download Newsletter", data=newsletter_content.encode("utf-8"), file_name=f"AI_Newsletter_{datetime.date.today()}.md", mime="text/markdown")
        
