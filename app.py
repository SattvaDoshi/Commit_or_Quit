import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import re
from wordcloud import WordCloud
import ast

# Set page config
st.set_page_config(
    page_title="Event Analyzer & Recommender",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #26A69A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Event Analyzer & Recommender</div>', unsafe_allow_html=True)
st.write("Explore event trends, recommend personalized events, and generate new event ideas.")

# File upload section
st.sidebar.header("Upload Data")
events_file = st.sidebar.file_uploader("Upload Events Dataset", type=["csv"])
users_file = st.sidebar.file_uploader("Upload Users Dataset", type=["csv"])
user_searches_file = st.sidebar.file_uploader("Upload User Searches (optional)", type=["csv"])
event_reviews_file = st.sidebar.file_uploader("Upload Event Reviews (optional)", type=["csv"])
social_media_file = st.sidebar.file_uploader("Upload Social Media Mentions (optional)", type=["csv"])

def safe_eval(val):
    """Safely evaluate string representations of lists or tuples"""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    return val

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(events_file, users_file):
    if events_file is not None and users_file is not None:
        events_df = pd.read_csv(events_file)
        users_df = pd.read_csv(users_file)
        
        # Ensure preferences is in the right format
        users_df["preferences"] = users_df["preferences"].apply(safe_eval)
        users_df["social_connections"] = users_df["social_connections"].apply(safe_eval)
        users_df["location"] = users_df["location"].apply(safe_eval)
        
        # Preprocess user preferences
        mlb = MultiLabelBinarizer()
        user_prefs = pd.DataFrame(mlb.fit_transform(users_df["preferences"]), 
                                  columns=mlb.classes_, index=users_df.index)
        user_features = pd.concat([users_df[["bookings"]], user_prefs], axis=1)
        
        # Normalize features before clustering
        scaler = StandardScaler()
        user_features_scaled = scaler.fit_transform(user_features)
        
        # Cluster users with optimized features
        kmeans = KMeans(n_clusters=5, random_state=42)
        users_df["cluster"] = kmeans.fit_predict(user_features_scaled)
        
        return events_df, users_df
    return None, None

@st.cache_data
def load_mock_data(events_df, users_df, user_searches_file, event_reviews_file, social_media_file):
    if user_searches_file is not None:
        user_searches = pd.read_csv(user_searches_file)
    else:
        # Create mock user searches data
        search_terms = [
            "outdoor yoga festival", "tech conference", "food truck festival",
            "virtual reality gaming", "sustainability workshop", "hybrid conference",
            "pet adoption event", "wellness retreat", "night market", "silent disco",
            "outdoor cinema", "craft beer tasting", "comedy night", "painting workshop",
            "cooking class", "book club meeting", "networking mixer", "charity run",
            "DIY crafting", "mindfulness meditation", "pop-up shop", "farmers market",
            "cultural festival", "escape room challenge", "urban gardening", "dance class"
        ]
        cities = events_df["city"].unique()
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

        n_searches = 5000
        user_searches = pd.DataFrame({
            "search_id": range(1, n_searches + 1),
            "user_id": np.random.choice(users_df["user_id"], n_searches),
            "search_term": np.random.choice(search_terms, n_searches),
            "city": np.random.choice(cities, n_searches),
            "search_date": np.random.choice(dates, n_searches),
            "converted_to_booking": np.random.choice([0, 1], n_searches, p=[0.8, 0.2])
        })
    
    if event_reviews_file is not None:
        event_reviews = pd.read_csv(event_reviews_file)
    else:
        # Create mock event reviews data
        n_reviews = 3000
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        event_reviews = pd.DataFrame({
            "review_id": range(1, n_reviews + 1),
            "event_id": np.random.choice(events_df["id"], n_reviews),
            "user_id": np.random.choice(users_df["user_id"], n_reviews),
            "rating": np.random.choice([1, 2, 3, 4, 5], n_reviews, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
            "sentiment_score": np.random.uniform(-1, 1, n_reviews),
            "review_date": np.random.choice(dates, n_reviews),
            "review_text": np.random.choice([
                "Great event, really enjoyed it!", "Was expecting more.",
                "Amazing experience!", "Not worth the price.",
                "Perfect venue.", "Loved the interaction.",
                "Too crowded.", "Well organized.",
                "Inspiring content.", "Highly recommend!"
            ], n_reviews)
        })
    
    if social_media_file is not None:
        social_media = pd.read_csv(social_media_file)
    else:
        # Create mock social media data
        n_mentions = 2000
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        social_media = pd.DataFrame({
            "mention_id": range(1, n_mentions + 1),
            "platform": np.random.choice(["Twitter", "Instagram", "Facebook", "TikTok"], n_mentions),
            "genre": np.random.choice(events_df["genre"].unique(), n_mentions),
            "mention_date": np.random.choice(dates, n_mentions),
            "engagement_score": np.random.uniform(0, 100, n_mentions),
            "sentiment_score": np.random.uniform(-1, 1, n_mentions),
            "region": np.random.choice(events_df["city"].unique(), n_mentions),
            "trending_hashtags": np.random.choice([
                "#eventideas #fun", "#weekend #activities", "#mustsee #entertainment",
                "#localhappenings #citylife", "#nightlife #experience",
                "#familyfun #memories", "#datenight #couples", "#uniqueexperiences #bucketlist"
            ], n_mentions)
        })
        
    # Make sure events_df has a bookings column
    if "bookings" not in events_df.columns:
        events_df["bookings"] = np.random.randint(0, 100, size=len(events_df))
        
    return user_searches, event_reviews, social_media, events_df

events_df, users_df = load_and_preprocess_data(events_file, users_file)

# Show placeholder message if data not uploaded
if events_df is None or users_df is None:
    st.info("Please upload both events and users datasets to get started.")
    st.stop()

# Load or generate mock data
user_searches, event_reviews, social_media, events_df = load_mock_data(
    events_df, users_df, user_searches_file, event_reviews_file, social_media_file)

# Calculate distance between user and event
def calculate_distance(user_loc, event_loc):
    return geodesic(user_loc, event_loc).miles

# Recommendation function
def recommend_events(user_id, top_n=5, exploration_factor=0.05):
    user = users_df[users_df["user_id"] == user_id].iloc[0]
    user_loc = user["location"]
    user_cluster = user["cluster"]
    user_prefs = user["preferences"]
    friends = user["social_connections"]

    # Filter nearby events first
    events_df["distance"] = events_df.apply(lambda row: calculate_distance(user_loc, (row["lat"], row["lon"])), axis=1)
    nearby_events = events_df[events_df["distance"] < 50].copy()

    # Improved scoring
    nearby_events["score"] = 0
    for pref in user_prefs:
        nearby_events["score"] += nearby_events["genre"].apply(lambda x: 1 if x == pref else 0) * 3.0  # Boost preference weight
    nearby_events["score"] += nearby_events["popularity"] * 0.5  # Moderate popularity weight
    nearby_events["score"] += np.exp(-nearby_events["distance"] / 50) * 1.0  # Exponential decay for distance

    # Friend bonus (only if friends are attending)
    friend_events = events_df[events_df["id"].isin(users_df[users_df["user_id"].isin(friends)].index)]["id"].tolist()
    nearby_events["friend_bonus"] = nearby_events["id"].apply(lambda x: 1.5 if x in friend_events else 0)
    nearby_events["score"] += nearby_events["friend_bonus"]

    # Controlled exploration: only include high-popularity events outside preferences
    exploration_events = nearby_events[~nearby_events["genre"].isin(user_prefs) & (nearby_events["popularity"] > 0.7)].sample(frac=exploration_factor, random_state=42)
    final_events = pd.concat([nearby_events, exploration_events]).drop_duplicates(subset="id")
    recommendations = final_events.sort_values(by="score", ascending=False).head(top_n)[["id", "name", "date", "venue", "city", "genre", "distance", "score"]]

    return recommendations

# 1. Trend Analysis
def analyze_emerging_trends(user_searches, social_media):
    top_search_terms = user_searches["search_term"].value_counts().head(10)
    
    user_searches["search_month"] = pd.to_datetime(user_searches["search_date"]).dt.strftime('%Y-%m')
    search_growth = user_searches.groupby(["search_month", "search_term"]).size().unstack().fillna(0)

    growth_rates = {}
    for term in search_growth.columns:
        term_data = search_growth[term].values
        if len(term_data) > 1 and term_data[0] > 0:
            growth_rate = (term_data[-1] - term_data[0]) / term_data[0]
            growth_rates[term] = growth_rate

    fastest_growing = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)[:5]
    
    platform_genre_counts = social_media.groupby(["platform", "genre"]).size().unstack().fillna(0)
    top_genre_by_platform = platform_genre_counts.idxmax(axis=1)
    
    all_hashtags = " ".join(social_media["trending_hashtags"].dropna())
    hashtags = re.findall(r'#(\w+)', all_hashtags)
    hashtag_counts = Counter(hashtags)
    
    return {
        "top_searches": top_search_terms.to_dict(),
        "growing_terms": dict(fastest_growing),
        "platform_trends": top_genre_by_platform.to_dict(),
        "hashtag_trends": dict(hashtag_counts.most_common(5))
    }

# 2. Regional Analysis
def analyze_regional_preferences(events_df, user_searches):
    event_region_popularity = events_df.groupby(["city", "genre"])["popularity"].mean().unstack().fillna(0)
    top_genre_by_region = event_region_popularity.idxmax(axis=1)
    
    unique_preferences = {}
    for region in event_region_popularity.index:
        regional_scores = event_region_popularity.loc[region]
        other_regions = event_region_popularity.drop(region)
        other_avg = other_regions.mean()
        difference = regional_scores - other_avg
        unique_preferences[region] = difference.sort_values(ascending=False).head(3)

    regional_searches = user_searches.groupby(["city", "search_term"]).size().unstack().fillna(0)
    regional_search_preferences = regional_searches.idxmax(axis=1)
    
    return {
        "regional_top_genres": top_genre_by_region.to_dict(),
        "unique_preferences": {region: prefs.to_dict() for region, prefs in unique_preferences.items()},
        "regional_searches": regional_search_preferences.to_dict()
    }

# 3. Market Demand Analysis
def analyze_market_demand(events_df, user_searches, event_reviews):
    event_bookings = events_df.groupby("genre")["bookings"].sum()
    event_counts = events_df["genre"].value_counts()
    avg_bookings_per_event = event_bookings / event_counts
    
    search_conversion = user_searches.groupby("search_term")["converted_to_booking"].mean().sort_values(ascending=False)
    
    search_counts = user_searches["search_term"].value_counts()
    search_terms_set = set(user_searches["search_term"].unique())
    search_genres = []
    for term in search_terms_set:
        words = term.lower().split()
        for genre in events_df["genre"].unique():
            if genre.lower() in words:
                search_genres.append((term, genre))
                break
        else:
            search_genres.append((term, "other"))

    search_to_genre = pd.DataFrame(search_genres, columns=["search_term", "mapped_genre"])
    search_to_genre = search_to_genre.merge(
        pd.DataFrame({"search_term": search_counts.index, "count": search_counts.values}),
        on="search_term"
    )

    genre_search_counts = search_to_genre.groupby("mapped_genre")["count"].sum()
    genre_event_counts = events_df["genre"].value_counts()

    genre_demand = {}
    for genre in set(genre_search_counts.index) | set(genre_event_counts.index):
        search_count = genre_search_counts.get(genre, 0)
        event_count = genre_event_counts.get(genre, 0)
        if event_count > 0:
            ratio = search_count / event_count
        else:
            ratio = search_count * 10  # High value for zero events
        genre_demand[genre] = ratio

    unmet_demand = sorted(genre_demand.items(), key=lambda x: x[1], reverse=True)
    
    genre_sentiment = event_reviews.merge(events_df[["id", "genre"]], left_on="event_id", right_on="id")
    genre_sentiment = genre_sentiment.groupby("genre")["sentiment_score"].mean().sort_values(ascending=False)
    
    return {
        "avg_bookings": avg_bookings_per_event.to_dict(),
        "search_conversion": search_conversion.to_dict(),
        "unmet_demand": dict(unmet_demand[:5]),
        "genre_sentiment": genre_sentiment.to_dict()
    }

# 4. Generate Event Ideas
def generate_event_ideas(trend_data, regional_data, market_data):
    ideas = []

    # Idea 1: Trending search terms + high-sentiment genres
    for search_term in trend_data["growing_terms"].keys():
        for genre, sentiment in market_data["genre_sentiment"].items():
            if sentiment > 0.5:
                words1 = set(search_term.lower().split())
                words2 = set(genre.lower().split())
                if len(words1 & words2) > 0 or len(words1) + len(words2) < 5:
                    ideas.append({
                        "name": f"{genre} {search_term}",
                        "description": f"Combine the growing trend of '{search_term}' with the highly rated '{genre}' category",
                        "target_regions": [],
                        "score": 0.8 + (0.2 * sentiment)
                    })

    # Idea 2: Address unmet demand
    for genre, ratio in market_data["unmet_demand"].items():
        if genre != "other" and ratio > 1:
            target_regions = [region for region, pref_genre in regional_data["regional_top_genres"].items() if pref_genre == genre]
            ideas.append({
                "name": f"Premium {genre} Experience",
                "description": f"High demand-to-supply ratio ({ratio:.1f}) indicates market opportunity for more {genre} events",
                "target_regions": target_regions,
                "score": min(1.0, 0.6 + (0.1 * ratio))
            })

    # Idea 3: Region-specific events
    for region, prefs in regional_data["unique_preferences"].items():
        top_unique_genres = sorted(prefs.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_unique_genres and top_unique_genres[0][1] > 0.2:
            genre1, score1 = top_unique_genres[0]
            if len(top_unique_genres) > 1:
                genre2, score2 = top_unique_genres[1]
                ideas.append({
                    "name": f"{region} {genre1} meets {genre2} Festival",
                    "description": f"Fusion event combining {region}'s two uniquely popular genres",
                    "target_regions": [region],
                    "score": 0.7 + (0.15 * score1)
                })
            else:
                ideas.append({
                    "name": f"{region} {genre1} Showcase",
                    "description": f"Celebrate {region}'s unique interest in {genre1}",
                    "target_regions": [region],
                    "score": 0.7 + (0.15 * score1)
                })

    # Idea 4: Social media platform trends
    for platform, genre in trend_data["platform_trends"].items():
        ideas.append({
            "name": f"{platform}-Inspired {genre} Event",
            "description": f"Create a {genre} event with strong {platform} integration to leverage platform popularity",
            "target_regions": [],
            "score": 0.75
        })

    # Idea 5: High conversion events
    for search_term, conv_rate in list(market_data["search_conversion"].items())[:3]:
        if conv_rate > 0.3:
            ideas.append({
                "name": f"Premium {search_term} Experience",
                "description": f"Search term with high booking conversion rate ({conv_rate*100:.1f}%)",
                "target_regions": [],
                "score": 0.6 + (0.4 * conv_rate)
            })

    return sorted(ideas, key=lambda x: x["score"], reverse=True)

# Create tabs for the application
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Event Recommendations", 
    "📈 Trend Analysis", 
    "🌍 Regional Analysis", 
    "📊 Market Demand", 
    "💡 Event Ideas"
])

# Tab 1: Event Recommendations
with tab1:
    st.markdown('<div class="section-header">Personalized Event Recommendations</div>', unsafe_allow_html=True)
    
    user_id_list = users_df["user_id"].unique().tolist()
    selected_user = st.selectbox("Select a user", user_id_list)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.slider("Number of recommendations", 3, 10, 5)
    with col2:
        exploration = st.slider("Exploration factor", 0.0, 0.2, 0.05, 0.01)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        recommend_btn = st.button("Get Recommendations")
    
    if recommend_btn or 'recommendations' in st.session_state:
        recommendations = recommend_events(selected_user, top_n, exploration)
        st.session_state.recommendations = recommendations
        
        st.markdown("### Recommended Events")
        
        # Format recommendations for display
        display_recs = recommendations.copy()
        display_recs["score"] = display_recs["score"].round(2)
        display_recs["distance"] = display_recs["distance"].round(1).astype(str) + " miles"
        
        # Display each recommendation as a card
        for i, rec in display_recs.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{rec['name']}**")
                st.write(f"📍 {rec['venue']}, {rec['city']} • 📅 {rec['date']} • 🏷️ {rec['genre']}")
            with col2:
                st.markdown(f"**Score: {rec['score']}**")
                st.write(f"Distance: {rec['distance']}")
            st.markdown("---")

# Tab 2: Trend Analysis
with tab2:
    st.markdown('<div class="section-header">Event Trend Analysis</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Trends") or 'trends' in st.session_state:
        trend_data = analyze_emerging_trends(user_searches, social_media)
        st.session_state.trends = trend_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Search Terms")
            fig, ax = plt.subplots(figsize=(10, 6))
            terms = list(trend_data["top_searches"].keys())[:8]
            counts = list(trend_data["top_searches"].values())[:8]
            ax.barh(terms, counts, color='skyblue')
            ax.set_xlabel('Search Count')
            ax.set_title('Top Event Search Terms')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            st.markdown("### Trending Hashtags")
            hashtags_df = pd.DataFrame({
                'Hashtag': [f"#{tag}" for tag in trend_data["hashtag_trends"].keys()],
                'Count': list(trend_data["hashtag_trends"].values())
            })
            st.table(hashtags_df)
            
        with col2:
            st.markdown("### Fastest Growing Search Terms")
            fig, ax = plt.subplots(figsize=(10, 6))
            terms = list(trend_data["growing_terms"].keys())
            growth = [g*100 for g in trend_data["growing_terms"].values()]
            ax.bar(terms, growth, color='lightgreen')
            ax.set_ylabel('Growth Rate (%)')
            ax.set_title('Fastest Growing Search Terms')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            st.markdown("### Trending Genres by Platform")
            platform_df = pd.DataFrame({
                'Platform': list(trend_data["platform_trends"].keys()),
                'Top Genre': list(trend_data["platform_trends"].values())
            })
            st.table(platform_df)
        
        # WordCloud of top search terms
        st.markdown("### Search Terms Word Cloud")
        search_text = " ".join([f"{term} " * count for term, count in trend_data["top_searches"].items()])
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, 
                              contour_width=1, contour_color='steelblue').generate(search_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# Tab 3: Regional Analysis
with tab3:
    st.markdown('<div class="section-header">Regional Event Preferences</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Regions") or 'regions' in st.session_state:
        regional_data = analyze_regional_preferences(events_df, user_searches)
        st.session_state.regions = regional_data
        
        # Create a DataFrame for the heat map
        event_region_popularity = events_df.groupby(["city", "genre"])["popularity"].mean().unstack().fillna(0)
        
        st.markdown("### Event Popularity Heatmap by Region")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(event_region_popularity, cmap="YlGnBu", annot=False, ax=ax)
        ax.set_title('Event Genre Popularity by Region')
        ax.set_ylabel('Region')
        ax.set_xlabel('Genre')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Genre by Region")
            top_genre_df = pd.DataFrame({
                'Region': list(regional_data["regional_top_genres"].keys()),
                'Top Genre': list(regional_data["regional_top_genres"].values())
            }).sort_values('Region')
            st.table(top_genre_df)
        
        with col2:
            st.markdown("### Most Searched Events by Region")
            search_pref_df = pd.DataFrame({
                'Region': list(regional_data["regional_searches"].keys()),
                'Most Searched': list(regional_data["regional_searches"].values())
            }).sort_values('Region')
            st.table(search_pref_df)
        
        # Region selection for unique preferences
        st.markdown("### Unique Regional Preferences")
        selected_region = st.selectbox("Select Region", list(regional_data["unique_preferences"].keys()))
        
        if selected_region:
            unique_prefs = regional_data["unique_preferences"][selected_region]
            unique_df = pd.DataFrame({
                'Genre': list(unique_prefs.keys()),
                'Difference from Average': list(unique_prefs.values())
            }).sort_values('Difference from Average', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(unique_df['Genre'], unique_df['Difference from Average'], 
                  color=['green' if x > 0 else 'red' for x in unique_df['Difference from Average']])
            ax.set_title(f'Unique Preferences for {selected_region}')
            ax.set_ylabel('Difference from Average Popularity')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.08,
                        f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top')
            
            st.pyplot(fig)

# Tab 4: Market Demand
with tab4:
    st.markdown('<div class="section-header">Market Demand Analysis</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Market") or 'market' in st.session_state:
        market_data = analyze_market_demand(events_df, user_searches, event_reviews)
        st.session_state.market = market_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Average Bookings per Event Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            genres = list(market_data["avg_bookings"].keys())[:8]
            bookings = list(market_data["avg_bookings"].values())[:8]
            ax.bar(genres, bookings, color='salmon')
            ax.set_ylabel('Average Bookings')
            ax.set_title('Average Bookings by Event Type')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            st.markdown("### Event Types with Highest Customer Satisfaction")
            sentiment_df = pd.DataFrame({
                'Genre': list(market_data["genre_sentiment"].keys())[:10],
                'Sentiment Score': [f"{score:.2f}" for score in list(market_data["genre_sentiment"].values())[:10]]
            })
            st.table(sentiment_df)
            
        with col2:
            st.markdown("### Search to Booking Conversion Rates")
            fig, ax = plt.subplots(figsize=(10, 6))
            terms = list(market_data["search_conversion"].keys())[:8]
            rates = [rate*100 for rate in list(market_data["search_conversion"].values())[:8]]
            ax.bar(terms, rates, color='lightgreen')
            ax.set_ylabel('Conversion Rate (%)')
            ax.set_title('Search Term Conversion Rates')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            st.markdown("### Event Types with Highest Demand-Supply Gap")
            unmet_df = pd.DataFrame({
                'Genre': [g for g in market_data["unmet_demand"].keys() if g != 'other'],
                'Demand-Supply Ratio': [f"{ratio:.2f}" for g, ratio in market_data["unmet_demand"].items() if g != 'other']
            })
            st.table(unmet_df)

# Tab 5: Event Ideas
with tab5:
    st.markdown('<div class="section-header">Generated Event Ideas</div>', unsafe_allow_html=True)
    
    if st.button("Generate Event Ideas") or 'event_ideas' in st.session_state:
        # Ensure previous analyses are run to provide data
        if 'trends' not in st.session_state:
            trend_data = analyze_emerging_trends(user_searches, social_media)
            st.session_state.trends = trend_data
        else:
            trend_data = st.session_state.trends
            
        if 'regions' not in st.session_state:
            regional_data = analyze_regional_preferences(events_df, user_searches)
            st.session_state.regions = regional_data
        else:
            regional_data = st.session_state.regions
            
        if 'market' not in st.session_state:
            market_data = analyze_market_demand(events_df, user_searches, event_reviews)
            st.session_state.market = market_data
        else:
            market_data = st.session_state.market
        
        event_ideas = generate_event_ideas(trend_data, regional_data, market_data)
        st.session_state.event_ideas = event_ideas
        
        # Display top event ideas
        st.markdown("### Top 10 Event Ideas")
        top_n_ideas = st.slider("Number of ideas to display", 5, min(20, len(event_ideas)), 10)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            for i, idea in enumerate(event_ideas[:top_n_ideas], 1):
                st.markdown(f"**{i}. {idea['name']}**")
                st.write(f"📝 {idea['description']}")
                if idea['target_regions']:
                    st.write(f"🎯 Target Regions: {', '.join(idea['target_regions'])}")
                st.markdown("---")
        with col2:
            # Bar chart of scores
            fig, ax = plt.subplots(figsize=(5, 6))
            idea_names = [idea['name'][:20] + "..." if len(idea['name']) > 20 else idea['name'] for idea in event_ideas[:top_n_ideas]]
            scores = [idea['score'] for idea in event_ideas[:top_n_ideas]]
            ax.barh(idea_names, scores, color='purple')
            ax.set_xlabel('Score')
            ax.set_title('Event Idea Scores')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        # Word Cloud of Event Idea Names
        st.markdown("### Event Ideas Word Cloud")
        idea_text = " ".join([f"{idea['name']} " * int(idea['score'] * 10) for idea in event_ideas[:top_n_ideas]])
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, 
                              contour_width=1, contour_color='purple').generate(idea_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Download button for event ideas
        ideas_df = pd.DataFrame(event_ideas)
        csv = ideas_df.to_csv(index=False)
        st.download_button(
            label="Download Event Ideas as CSV",
            data=csv,
            file_name="event_ideas.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    st.write("Application loaded successfully!")