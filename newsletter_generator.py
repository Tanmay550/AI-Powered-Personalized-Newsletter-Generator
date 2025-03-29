import streamlit as st
import feedparser
import torch
from transformers import pipeline
import datetime
import os
import re
from bs4 import BeautifulSoup

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
        "http://feeds.reuters.com/reuters/topNews",
        "https://www.business-standard.com/rss/companies/"
    ],
    "Technology": [
        "https://techcrunch.com/tag/rss/",
        "https://www.wired.com/feed/rss",
        "https://www.technologyreview.com/feed/",
        "https://tech.hindustantimes.com/rss/tech/news",
        "https://tech.hindustantimes.com/rss/tech"
    ],
    "Finance": [
        "https://www.bloomberg.com/feed/",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.ft.com/business-education?format=rss",
        "https://cfo.economictimes.indiatimes.com/rss/corporate-finance",
        "https://cfo.economictimes.indiatimes.com/rss/topstories",
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://cointelegraph.com/rss",
        "https://www.finextra.com/rss/channel.aspx?channel=startups",
        "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"
    ],
    "Sports": [
        "https://www.espn.com/espn/rss/news",
        "https://rss.app/feeds/tWwnqzFBv5TWuhb2.xml",
        "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/mlb",
        "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/nba",
        "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/soccer,soccer/epl/league/1,soccer/mls/league/5,soccer/ucl/league/7,soccer/europa/league/8,soccer/wc/league/12,soccer/euro/league/13,soccer/wwc/league/14,soccer/nwsl/league/20,soccer/cwc/league/26,soccer/gold_cup/league/32,soccer/unl/league/67",
        "https://moxie.foxnews.com/google-publisher/sports.xml",
        "https://www.reddit.com/r/RocketLeagueEsports/.rss",
        "https://www.autosport.com/rss/f1/news/",
        "https://www.nytimes.com/athletic/rss/news/"
    ],
    "Entertainment": [
        "https://variety.com/feed/",
        "https://www.hollywoodreporter.com/t/hollywood/feed/",
        "https://www.billboard.com/feed/",
        "https://www.usmagazine.com/category/entertainment/feed/",
        "https://www.etonline.com/news/rss/",
        "https://www.bookbrowse.com/rss/"
    ],
    "Science": [
        "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "https://www.sciencedaily.com/rss/all.xml",
        "https://feeds.arstechnica.com/arstechnica/science",
        " https://scipost.org/rss/news/",
        "https://phys.org/rss-feed/breaking/",
        "https://phys.org/rss-feed/breaking/physics-news/quantum-physics/",
        "https://phys.org/rss-feed/breaking/biology-news/biotechnology/",
        "https://www.wired.com/feed/tag/ai/latest/rss",
        "https://www.wired.com/feed/category/science/latest/rss"
    ]
}


def clean_html_content(html_text):
    if not html_text:
        return "No summary available"
    
    try:
        # Use BeautifulSoup to parse HTML and extract text
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        # If parsing fails, fallback to simple regex cleaning
        text = re.sub(r'<[^>]+>', ' ', html_text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    

def fetch_rss_articles(url):
    feed = feedparser.parse(url)
    articles = []
    
    for entry in feed.entries[:5]:
        # Clean the summary/content from HTML tags
        summary = entry.get("summary", entry.get("description", "No Summary"))
        clean_summary = clean_html_content(summary)
        
        articles.append({
            "title": entry.get("title", "No Title"),
            "summary": clean_summary,
            "link": entry.get("link", "No Link"),
            "source": feed.feed.get("title", url),
            "published": entry.get("published", "Unknown date")
        })
    
    return articles if articles else []

# Classify Articles into User Categories - Using improved confidence threshold
def classify_article(text, user_categories):
    text = text[:512]  # Limit text length for better classification
    try:
        result = classifier(text, user_categories, multi_label=True)
        # Return the category with highest score above threshold
        top_category = result["labels"][0]
        top_score = result["scores"][0]
        if top_score > 0.5:  # Increased confidence threshold from 0.3 to 0.5
            return top_category
    except Exception:
        return None
    return None

# Improved feed source selection - Direct match with category names
def determine_feed_sources(user_categories):
    sources = []
    
    # First, check for direct matches with RSS feed categories
    for category in user_categories:
        if category in RSS_FEEDS:
            sources.append(category)
        else:
            # If no direct match, use classification to find best feed source
            try:
                result = classifier(category, list(RSS_FEEDS.keys()), multi_label=False)
                top_source = result["labels"][0]
                top_score = result["scores"][0]
                
                # Only add if confidence is high
                if top_score > 0.4:  # Increased threshold
                    sources.append(top_source)
                else:
                    # If no good match, only add General News if explicitly requested
                    if category.lower() in ["news", "general news", "current events"]:
                        sources.append("General News")
            except Exception:
                # No fallbacks to avoid unwanted content
                pass
    
    # If no sources found at all, add General News as fallback
    if not sources:
        sources.append("General News")
    
    return list(set(sources))  # Remove duplicates

# Get feeds based on selected source categories
def get_feeds_for_sources(source_categories):
    all_feeds = []
    
    for category in source_categories:
        if category in RSS_FEEDS:
            all_feeds.extend(RSS_FEEDS[category])
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(all_feeds))

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
    # Extract all summaries
    all_summaries = " ".join([summarize_text(article["summary"]) for articles in articles_by_category.values() for article in articles])

    # Generate an overall summary of all articles
    general_summary = ""
    if len(all_summaries.split()) > 50:
        try:
            general_summary = summarizer(all_summaries, max_length=250, min_length=100, do_sample=False)[0]["summary_text"]
        except Exception:
            general_summary = "Summary not available due to processing error."

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
st.title("📢 AI-Powered Custom News Categorization")
st.sidebar.header("User Preferences")


# Text area for user to enter categories
custom_categories_input = st.sidebar.text_area(
    "Enter your categories (one per line):",
    height=150,
    help="Examples: Artificial Intelligence, Climate Change, Politics, Space Exploration, etc."
)

# Parse custom categories
user_categories = [category.strip() for category in custom_categories_input.split('\n') if category.strip()]
user_categories = list(dict.fromkeys(user_categories))  # Remove duplicates

# Main fetch button
if st.sidebar.button("Fetch News"):
    if not user_categories:
        st.warning("⚠️ Please define at least one category of interest.")
    else:
        # Determine which feed sources to use based on user categories
        feed_sources = determine_feed_sources(user_categories)
        all_feeds = get_feeds_for_sources(feed_sources)
        
        st.subheader("📰 AI-Categorized News Based on Your Interests")
        st.info(f"Using sources from: {', '.join(feed_sources)}")
        
        # Create articles dictionary by category
        articles_by_category = {category: [] for category in user_categories}
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_feeds = len(all_feeds)
        if total_feeds == 0:
            st.warning("No feeds found matching your categories. Try different categories.")
        else:
            # Process each feed
            for i, url in enumerate(all_feeds):
                status_text.text(f"Processing feed {i+1} of {total_feeds}...")
                articles = fetch_rss_articles(url)
                
                for article in articles:
                    # Combine title and summary for better classification
                    text = f"{article['title']} {article['summary']}"
                    classified_category = classify_article(text, user_categories)
                    
                    if classified_category:
                        # Generate a summarized version
                        article["summary"] = summarize_text(article["summary"])
                        # Add to the appropriate category
                        articles_by_category[classified_category].append(article)
                
                # Update progress
                progress_bar.progress((i + 1) / total_feeds)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            empty_categories = True
            for category, articles in articles_by_category.items():
                if articles:
                    empty_categories = False
                    st.markdown(f"## 📌 {category}")
                    for article in articles:
                        st.markdown(f"### 🔍 {article['title']}")
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"📄 **Summary:** {article['summary']}")
                        st.markdown(f"[Read more...]({article['link']})")
                        st.write("---")
            
            if empty_categories:
                st.warning("No articles matching your categories were found. Try adding more diverse categories.")
            else:
                # Generate downloadable newsletter
                newsletter_content = generate_newsletter(articles_by_category)
                st.download_button(
                    label="📩 Download Newsletter", 
                    data=newsletter_content.encode("utf-8"), 
                    file_name=f"Custom_AI_Newsletter_{datetime.date.today()}.md", 
                    mime="text/markdown"
                )
else:
    # Show instructions when app first loads
    st.markdown("""
    ## 🌟 How to Use
    1. **Enter your categories** of interest in the sidebar (one per line)
    2. Click **Fetch News** to get AI-categorized articles matching your interests
    3. The app will automatically select the appropriate news sources based on your categories
    4. Download a personalized newsletter with your curated content
    
    Some suggested categories include Technology, Finance, Science, F1, Esports, Basketball, Soccer, Baseball, Entertainment, Sports, and General News for more accurate results.
    """)