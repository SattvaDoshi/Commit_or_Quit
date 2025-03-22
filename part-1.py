import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sparse
import gc
import pickle
import os
from tqdm.notebook import tqdm
import time

# ============= CONFIGURATION =============
# Set these parameters to control memory usage and speed
MAX_ROWS = None  # Set to a number to limit data rows (e.g., 50000)
REDUCED_FEATURES = 30  # Reduced dimensionality for embeddings
USE_CACHED_DATA = True  # Set to False to regenerate matrices/embeddings
CHUNK_SIZE = 10000  # Process data in chunks of this size
MAX_FRIENDS = 50  # Limit number of friends to process
TOP_K_SIMILAR_USERS = 30  # Only consider this many similar users
SAMPLE_EVAL_SIZE = 200  # Number of samples for evaluation
LOCATION_BOOST = 0.5  # Increased weight for location match (0-1)
# =========================================

def log_memory_usage(message):
    """Log memory usage at different points in the code"""
    import psutil
    memory = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"{message} - Memory usage: {memory:.2f} MB")

# Memory-saving function to load data with optimized dtypes
def load_data(file_path, usecols=None, nrows=MAX_ROWS):
    print(f"Loading {file_path.split('/')[-1]}...")
    start_time = time.time()

    # First pass to determine column types
    dtypes = {}
    if usecols is None:
        df_sample = pd.read_csv(file_path, nrows=1000)
        usecols = df_sample.columns.tolist()

    # Determine optimal dtypes to save memory
    for col in usecols:
        if 'id' in col.lower() or col in ['user', 'event']:
            dtypes[col] = 'str'  # Use string type for IDs to ensure consistency

    # Load the data with optimized dtypes
    df = pd.read_csv(file_path, usecols=usecols, nrows=nrows, dtype=dtypes)

    # Convert object columns with few unique values to category
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['user', 'event', 'user_id', 'event_id']:  # Skip ID columns
            if df[col].nunique() < df.shape[0] * 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return df

# Create cache directory if it doesn't exist
os.makedirs('/content/drive/MyDrive/D2K/cache', exist_ok=True)
cache_dir = '/content/drive/MyDrive/D2K/cache'

# Function to save/load cached data
def get_cached_data(filename, data_generator_func, force_refresh=not USE_CACHED_DATA):
    """Get data from cache or generate it if not available"""
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached {filename}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Generating {filename}...")
        data = data_generator_func()
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return data

# Load only essential columns from each dataset
print("Loading datasets with optimized memory usage...")

# Update column selection to include location data
events_cols = ['event_id', 'city', 'country']  # Added city and country
users_cols = ['user_id', 'location']           # Added location
user_friends_cols = ['user', 'friends']
train_cols = ['user', 'event', 'interested']
event_attendees_cols = ['event', 'attendee']  # Add only if needed

# Load data with memory optimization
events = load_data("/content/drive/MyDrive/d2k/events.csv", usecols=events_cols)
users = load_data("/content/drive/MyDrive/d2k/users.csv", usecols=users_cols)
user_friends = load_data("/content/drive/MyDrive/d2k/user_friends.csv", usecols=user_friends_cols)
train = load_data("/content/drive/MyDrive/d2k/train.csv", usecols=train_cols)

# Optionally load event_attendees only if needed
if 'event_attendees' in locals():
    event_attendees = load_data("/content/drive/MyDrive/d2k/event_attendees.csv", usecols=event_attendees_cols)

log_memory_usage("After loading datasets")

# Ensure consistent data types for all IDs
def ensure_consistent_types():
    """Convert all IDs to strings for consistent comparison"""
    # Convert user and event IDs to strings
    train['user'] = train['user'].astype(str)
    train['event'] = train['event'].astype(str)
    events['event_id'] = events['event_id'].astype(str)
    users['user_id'] = users['user_id'].astype(str)
    user_friends['user'] = user_friends['user'].astype(str)

    # If friends column is string, make sure to convert each friend ID in the list too
    if user_friends['friends'].dtype == 'object':
        # We'll handle the conversion when processing friends
        pass

    print("Data types converted to consistent format")

# Run type conversion
ensure_consistent_types()

# Process location data efficiently with improved error handling
def process_location_data():
    """Extract and normalize location data from users and events"""
    print("Processing location data...")

    # Process user locations
    user_locations = {}

    print("Processing user locations...")
    for _, row in tqdm(users.iterrows(), total=len(users)):
        user_id = str(row['user_id'])
        location = str(row['location']) if pd.notna(row['location']) and row['location'] != '' else ""

        # Skip empty locations
        if not location.strip():
            continue

        # Extract city and country
        parts = [part.strip() for part in location.split(',')]

        if len(parts) >= 2:
            city, country = parts[0].lower(), parts[-1].lower()
        elif len(parts) == 1:
            city, country = parts[0].lower(), ""
        else:
            continue  # Skip if no valid location components

        # Print some examples to debug
        if len(user_locations) < 5:
            print(f"Example: User {user_id} location: '{location}' -> city: '{city}', country: '{country}'")

        user_locations[user_id] = {'city': city, 'country': country}

    # Process event locations
    event_locations = {}

    print("Processing event locations...")
    for _, row in tqdm(events.iterrows(), total=len(events)):
        event_id = str(row['event_id'])
        city = str(row['city']).lower() if pd.notna(row['city']) and row['city'] != '' else ""
        country = str(row['country']).lower() if pd.notna(row['country']) and row['country'] != '' else ""

        # Skip empty locations
        if not city and not country:
            continue

        # Print some examples to debug
        if len(event_locations) < 5:
            print(f"Example: Event {event_id} location: city: '{city}', country: '{country}'")

        event_locations[event_id] = {'city': city, 'country': country}

    print(f"Processed {len(user_locations)} user locations and {len(event_locations)} event locations")
    return user_locations, event_locations

# Load or generate location data - force refresh to ensure proper processing
location_data = get_cached_data(
    'location_data.pkl',
    process_location_data,
    force_refresh=True  # Always refresh location data to ensure proper processing
)

user_locations, event_locations = location_data
log_memory_usage("After processing location data")

# Print location data stats
print(f"Number of users with location data: {len(user_locations)}")
print(f"Number of events with location data: {len(event_locations)}")
if user_locations:
    print("Sample user locations:", list(user_locations.items())[:3])
if event_locations:
    print("Sample event locations:", list(event_locations.items())[:3])

# Make sure numeric columns use appropriate types
for col in train.select_dtypes(include=['int64']).columns:
    if train[col].min() >= 0:
        if train[col].max() < 255:
            train[col] = train[col].astype(np.uint8)
        elif train[col].max() < 65535:
            train[col] = train[col].astype(np.uint16)
        elif train[col].max() < 4294967295:
            train[col] = train[col].astype(np.uint32)

# Fill missing values efficiently
train['interested'] = train['interested'].fillna(0).astype(np.uint8)

log_memory_usage("After preprocessing train data")

# Get event description columns if they exist
event_text_cols = [col for col in events.columns if col not in ['event_id', 'city', 'country']]

# Generate event embeddings with reduced dimensions and caching
def generate_event_embeddings():
    if len(event_text_cols) > 0:
        # Process text efficiently
        print("Processing event text...")
        event_text = events[event_text_cols].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)

        # Use TF-IDF with fewer features
        print("Calculating TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=REDUCED_FEATURES, stop_words='english')
        event_desc_matrix = vectorizer.fit_transform(event_text)

        # Use SVD with fewer components
        print("Performing SVD...")
        svd = TruncatedSVD(n_components=REDUCED_FEATURES, random_state=42)
        embeddings = svd.fit_transform(event_desc_matrix)

        # Create mapping
        id_to_idx = {event_id: idx for idx, event_id in enumerate(events['event_id'])}

        # Clean up to save memory
        del event_text, event_desc_matrix, vectorizer, svd
        gc.collect()

        return embeddings, id_to_idx
    else:
        # If no text columns, create dummy embeddings
        print("No text columns found, creating dummy embeddings...")
        dummy_embeddings = np.random.random((len(events), REDUCED_FEATURES))
        id_to_idx = {event_id: idx for idx, event_id in enumerate(events['event_id'])}
        return dummy_embeddings, id_to_idx

# Try to load cached embeddings or generate them
event_embeddings, event_id_to_idx = get_cached_data(
    'event_embeddings.pkl',
    generate_event_embeddings
)

log_memory_usage("After generating embeddings")

# Optimize the "get_embedding" function
def get_event_embedding(event_id):
    """Efficiently get vector representation of an event"""
    idx = event_id_to_idx.get(str(event_id))
    if idx is not None:
        return event_embeddings[idx]
    return np.zeros(REDUCED_FEATURES)  # Return zero vector if event not found

# Process user social connections more efficiently
def process_social_connections():
    """Process social connections with memory optimization"""
    social_map = {}
    print("Processing friend connections...")
    for chunk_start in tqdm(range(0, len(user_friends), CHUNK_SIZE)):
        chunk = user_friends.iloc[chunk_start:chunk_start+CHUNK_SIZE]

        for _, row in chunk.iterrows():
            user = str(row['user'])
            if isinstance(row['friends'], str) and row['friends']:
                # Convert friend IDs to strings for consistency
                friends = [str(friend) for friend in row['friends'].split()[:MAX_FRIENDS]]
                social_map[user] = set(friends)

        # Clear chunk to free memory
        del chunk

    return social_map

# Load or generate social map
user_social_map = get_cached_data(
    'user_social_map.pkl',
    process_social_connections
)

# Free memory
del user_friends
gc.collect()
log_memory_usage("After processing social connections")

def get_social_influence(user_id):
    """Get social influence with optimized processing"""
    user_id = str(user_id)  # Ensure string type
    friends = user_social_map.get(user_id, set())
    if not friends:
        return set()

    # Use more efficient filtering
    attended_mask = (train['interested'] == 1)
    friend_mask = train['user'].isin(friends)
    attended_events = train[attended_mask & friend_mask]['event'].unique()

    return set(attended_events)

# Improved function to calculate location similarity
def calculate_location_similarity(user_id, event_id):
    """Calculate location similarity score between a user and an event"""
    user_id = str(user_id)  # Ensure string type
    event_id = str(event_id)  # Ensure string type

    user_loc = user_locations.get(user_id, {'city': '', 'country': ''})
    event_loc = event_locations.get(event_id, {'city': '', 'country': ''})

    # Initialize score
    score = 0.0

    # Check for city match (higher weight)
    if user_loc['city'] and event_loc['city'] and user_loc['city'] == event_loc['city']:
        score += 0.7

    # Check for country match
    if user_loc['country'] and event_loc['country'] and user_loc['country'] == event_loc['country']:
        score += 0.3

    return score

# Create optimized interaction matrix with chunking and caching
def create_interaction_matrix_optimized():
    """Create sparse interaction matrix with memory optimization"""
    # Get unique users and items - convert to list for faster lookups
    users = train['user'].unique()
    items = train['event'].unique()

    # Create mappings using dictionaries
    user_map = {user: idx for idx, user in enumerate(users)}
    item_map = {item: idx for idx, item in enumerate(items)}

    # Create sparse matrix with chunking
    rows, cols, data = [], [], []

    print("Creating interaction matrix...")
    for chunk_start in tqdm(range(0, len(train), CHUNK_SIZE)):
        chunk = train.iloc[chunk_start:chunk_start+CHUNK_SIZE]

        chunk_rows = [user_map[user] for user in chunk['user']]
        chunk_cols = [item_map[item] for item in chunk['event']]
        chunk_data = [int(rating) for rating in chunk['interested']]

        rows.extend(chunk_rows)
        cols.extend(chunk_cols)
        data.extend(chunk_data)

        # Clear chunk to free memory
        del chunk, chunk_rows, chunk_cols, chunk_data
        gc.collect()

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    # Create reverse mappings
    user_map_reversed = {idx: user for user, idx in user_map.items()}
    item_map_reversed = {idx: item for item, idx in item_map.items()}

    return matrix, user_map, item_map, user_map_reversed, item_map_reversed

# Load or create interaction matrix
matrix_data = get_cached_data(
    'interaction_matrix_data.pkl',
    create_interaction_matrix_optimized
)

user_item_matrix, user_map, item_map, user_map_reversed, item_map_reversed = matrix_data
log_memory_usage("After creating interaction matrix")

# Optimized similarity calculation for a specific user
def calculate_similarity_for_user(user_idx, top_k=TOP_K_SIMILAR_USERS):
    """Calculate similarity for a specific user only"""
    user_vector = user_item_matrix[user_idx]

    # Calculate similarity with all users - one row at a time to save memory
    similarities = []
    for i in range(user_item_matrix.shape[0]):
        if i != user_idx:  # Skip self
            other_vector = user_item_matrix[i]
            # Use dot product for sparse vectors
            sim = user_vector.dot(other_vector.T).toarray()[0][0]
            # Normalize by vector lengths if non-zero
            norm_product = np.sqrt((user_vector.power(2)).sum() * (other_vector.power(2)).sum())
            if norm_product > 0:
                sim = sim / norm_product
            similarities.append((i, sim))

    # Sort and get top k
    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    similar_indices = [idx for idx, _ in top_similar]
    similar_scores = [score for _, score in top_similar]

    return similar_indices, similar_scores

# Improved recommendation function with better location handling
def recommend_items_for_user(user_id, num_recommendations=5):
    """Generate recommendations for a user efficiently with improved location handling"""
    user_id = str(user_id)  # Ensure string type

    if user_id not in user_map:
        print(f"User {user_id} not found in user_map")
        return pd.DataFrame(columns=['event_id', 'predicted_interest', 'location_score'])

    user_idx = user_map[user_id]
    user_interactions = user_item_matrix[user_idx].toarray().reshape(-1)

    # Get similar users
    similar_user_indices, similar_user_scores = calculate_similarity_for_user(user_idx)

    if not similar_user_indices:
        print(f"No similar users found for {user_id}")
        return pd.DataFrame(columns=['event_id', 'predicted_interest', 'location_score'])

    # Calculate weighted scores
    weighted_scores = np.zeros(user_item_matrix.shape[1])

    for i, other_user in enumerate(similar_user_indices):
        sim_score = similar_user_scores[i]
        if sim_score > 0:
            other_interactions = user_item_matrix[other_user].toarray().reshape(-1)
            weighted_scores += sim_score * other_interactions

    # Filter out already interacted items
    already_interacted = user_interactions.nonzero()[0]
    weighted_scores[already_interacted] = 0

    # Get top N*2 candidates (will filter by location later)
    top_candidates_idx = weighted_scores.argsort()[-(num_recommendations*2):][::-1]
    top_candidates = [item_map_reversed[idx] for idx in top_candidates_idx]
    top_scores = weighted_scores[top_candidates_idx]

    # Calculate location scores and adjust final scores
    location_scores = []
    adjusted_scores = []

    # Check if user has location data
    has_location = user_id in user_locations

    for i, event_id in enumerate(top_candidates):
        # Calculate location similarity score
        loc_score = calculate_location_similarity(user_id, event_id)
        location_scores.append(loc_score)

        # Adjust score with location boost
        adjusted_scores.append(top_scores[i] * (1 + LOCATION_BOOST * loc_score))

    # Create DataFrame with all scores
    recommendations_df = pd.DataFrame({
        'event_id': top_candidates,
        'cf_score': top_scores,
        'location_score': location_scores,
        'adjusted_score': adjusted_scores
    })

    # Sort by adjusted score and take top N
    recommendations_df = recommendations_df.sort_values('adjusted_score', ascending=False).head(num_recommendations)
    recommendations_df = recommendations_df.rename(columns={'adjusted_score': 'predicted_interest'})

    return recommendations_df[['event_id', 'predicted_interest', 'location_score']]

# Enhanced recommendation function with stronger location preference
def recommend_events(user_id, num_recommendations=5):
    """Main recommendation function with improved location-based recommendations"""
    user_id = str(user_id)  # Ensure string type

    try:
        # Check if user has location data
        has_location = user_id in user_locations

        # Start with collaborative filtering with location boost
        recommendations = recommend_items_for_user(user_id, num_recommendations)

        # If user has location, boost the weight of location even more
        if has_location:
            # Re-weight scores to emphasize location more
            recommendations['predicted_interest'] = recommendations['predicted_interest'] * (1 + recommendations['location_score'])
            recommendations = recommendations.sort_values('predicted_interest', ascending=False)

        # If we have enough recommendations, return them
        if len(recommendations) >= num_recommendations:
            return recommendations

        # If we need more, try content-based approach
        if len(recommendations) < num_recommendations:
            # Get events the user liked
            user_events = train[(train['user'] == user_id) & (train['interested'] == 1)]['event'].tolist()[:10]

            if user_events:
                # Get embeddings and calculate content-based recommendations
                user_embeddings = [get_event_embedding(event_id) for event_id in user_events]
                user_profile = np.mean(user_embeddings, axis=0)

                # Find similar events with location boost
                content_scores = []

                # Only check a limited number of events to save time
                candidate_events = list(train['event'].unique())[:1000]  # Limit to first 1000 events

                for event_id in candidate_events:
                    if event_id not in user_events and event_id not in recommendations['event_id'].values:
                        event_embedding = get_event_embedding(event_id)
                        sim = np.dot(user_profile, event_embedding) / (np.linalg.norm(user_profile) * np.linalg.norm(event_embedding) + 1e-8)

                        # Calculate location score
                        loc_score = calculate_location_similarity(user_id, event_id)

                        # Adjust similarity score with location boost - use higher weight
                        adjusted_sim = sim * (1 + LOCATION_BOOST * 2 * loc_score)

                        content_scores.append((event_id, adjusted_sim, loc_score))

                # Sort and get top content-based recommendations
                content_recs = sorted(content_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]

                # Add these to our recommendations
                for event_id, score, loc_score in content_recs:
                    if event_id not in recommendations['event_id'].values:
                        new_row = pd.DataFrame({
                            'event_id': [event_id],
                            'predicted_interest': [score],
                            'location_score': [loc_score]
                        })
                        recommendations = pd.concat([recommendations, new_row], ignore_index=True)

                        if len(recommendations) >= num_recommendations:
                            break

        # Final sort by predicted interest and return top N
        recommendations = recommendations.sort_values('predicted_interest', ascending=False).head(num_recommendations)
        return recommendations

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        # Return empty dataframe if there's an error
        return pd.DataFrame(columns=['event_id', 'predicted_interest', 'location_score'])

# Alternative recommendation approach with strong location priority
def recommend_events_with_location_priority(user_id, num_recommendations=5):
    """Alternative recommendation approach that prioritizes location matching"""
    user_id = str(user_id)  # Ensure string type

    # Get user location
    user_loc = user_locations.get(user_id, {'city': '', 'country': ''})

    if not user_loc['city'] and not user_loc['country']:
        # If user has no location data, use regular recommendations
        return recommend_events(user_id, num_recommendations)

    print(f"User {user_id} location: {user_loc}")

    # 1. First, find events in the same city
    city_events = []
    if user_loc['city']:
        for event_id, loc in event_locations.items():
            if loc['city'] == user_loc['city']:
                city_events.append(event_id)

    # 2. Then, find events in the same country
    country_events = []
    if user_loc['country']:
        for event_id, loc in event_locations.items():
            if loc['country'] == user_loc['country'] and event_id not in city_events:
                country_events.append(event_id)

    print(f"Found {len(city_events)} events in same city and {len(country_events)} events in same country")

    # Get standard recommendations
    std_recs = recommend_events(user_id, num_recommendations * 2)  # Get more recommendations than needed

    # Create a combined recommendation set
    combined_recs = []

    # First add recommendations that are in the same city
    for _, row in std_recs.iterrows():
        event_id = row['event_id']
        if event_id in city_events:
            combined_recs.append({
                'event_id': event_id,
                'predicted_interest': row['predicted_interest'] * 2,  # Boost city matches
                'location_score': 1.0  # Maximum location score
            })

    # Then add recommendations that are in the same country
    for _, row in std_recs.iterrows():
        event_id = row['event_id']
        if event_id in country_events and len(combined_recs) < num_recommendations:
            combined_recs.append({
                'event_id': event_id,
                'predicted_interest': row['predicted_interest'] * 1.5,  # Boost country matches
                'location_score': 0.5  # Medium location score
            })

    # Fill remaining slots with standard recommendations
    for _, row in std_recs.iterrows():
        event_id = row['event_id']
        if event_id not in city_events and event_id not in country_events and len(combined_recs) < num_recommendations:
            combined_recs.append({
                'event_id': event_id,
                'predicted_interest': row['predicted_interest'],
                'location_score': row['location_score']
            })

    return pd.DataFrame(combined_recs).head(num_recommendations)

# Test function for location-based recommendations
def test_location_recommendations():
    """Test location-based recommendations for a specific user"""
    # Find a user with location data
    for user_id in list(user_locations.keys())[:100]:
        if user_id in user_map:
            print(f"\nTesting recommendations for user {user_id}")
            print(f"User location: {user_locations[user_id]}")

            # Get recommendations
            recs = recommend_events(user_id, 5)
            print("\nRecommendations:")
            print(recs)

            # Check each recommended event's location
            print("\nRecommended event locations:")
            for event_id in recs['event_id']:
                loc = event_locations.get(event_id, {'city': 'Unknown', 'country': 'Unknown'})
                print(f"Event {event_id}: {loc}")

            # Also try the location priority method
            print("\nRecommendations with location priority:")
            loc_recs = recommend_events_with_location_priority(user_id, 5)
            print(loc_recs)

            break
    else:
        print("No suitable test user found")

# Enhanced evaluation function with location metrics
def evaluate_recommendations(num_samples=SAMPLE_EVAL_SIZE):
    """Evaluate recommendations with metrics for both interest and location relevance"""
    print(f"Running evaluation on {num_samples} samples...")

    # Get a small sample of users
    sample_users = list(user_map.keys())[:num_samples]

    # Metrics
    interest_hits = 0
    location_hits = 0
    total = 0

    for user in tqdm(sample_users):
        # Get actual events the user liked (from the data)
        actual_liked = set(train[(train['user'] == user) & (train['interested'] == 1)]['event'].tolist())
        if not actual_liked:
            continue

        # Get recommendations
        recs = recommend_events(user, num_recommendations=10)
        recommended_events = set(recs['event_id'].tolist())

        # Count interest hits (recommended events that the user actually liked)
        interest_common = recommended_events.intersection(actual_liked)
        interest_hits += len(interest_common)

        # Count location hits (recommended events in the same city/country as user)
        for event_id in recommended_events:
            if calculate_location_similarity(user, event_id) > 0:
                location_hits += 1

        total += min(len(actual_liked), 10)  # Count at most 10 possible hits

    # Calculate hit rates
    interest_hit_rate = interest_hits / total if total > 0 else 0
    location_hit_rate = location_hits / (len(sample_users) * 10) if sample_users else 0

    print(f"Interest hit rate: {interest_hit_rate:.4f} ({interest_hits}/{total})")
    print(f"Location relevance rate: {location_hit_rate:.4f} ({location_hits}/{len(sample_users)*10})")

    return interest_hit_rate, location_hit_rate

# Also evaluate the location priority approach
def compare_recommendation_approaches(num_samples=50):
    """Compare standard vs location-priority recommendations"""
    print(f"Comparing recommendation approaches on {num_samples} samples...")

    # Get users with location data
    users_with_location = [u for u in list(user_map.keys())[:num_samples*5] if u in user_locations]
    sample_users = users_with_location[:num_samples]

    if not sample_users:
        print("No users with location data found for comparison")
        return

    print(f"Found {len(sample_users)} users with location data")

    # Metrics for standard approach
    std_interest_hits = 0
    std_location_hits = 0

    # Metrics for location priority approach
    loc_interest_hits = 0
    loc_location_hits = 0

    total = 0

    for user in tqdm(sample_users):
        # Get actual events the user liked (from the data)
        actual_liked = set(train[(train['user'] == user) & (train['interested'] == 1)]['event'].tolist())
        if not actual_liked:
            continue

        # Get recommendations from both approaches
        std_recs = recommend_events(user, num_recommendations=5)
        loc_recs = recommend_events_with_location_priority(user, num_recommendations=5)

        std_events = set(std_recs['event_id'].tolist())
        loc_events = set(loc_recs['event_id'].tolist())

        # Count interest hits for standard approach
        std_interest_common = std_events.intersection(actual_liked)
        std_interest_hits += len(std_interest_common)

        # Count location hits for standard approach
        for event_id in std_events:
            if calculate_location_similarity(user, event_id) > 0:
                std_location_hits += 1

        # Count interest hits for location priority approach
        loc_interest_common = loc_events.intersection(actual_liked)
        loc_interest_hits += len(loc_interest_common)

        # Count location hits for location priority approach
        for event_id in loc_events:
            if calculate_location_similarity(user, event_id) > 0:
                loc_location_hits += 1

        total += min(len(actual_liked), 5)  # Count at most 5 possible hits

    # Calculate hit rates
    std_interest_hit_rate = std_interest_hits / total if total > 0 else 0
    std_location_hit_rate = std_location_hits / (len(sample_users) * 5) if sample_users else 0

    loc_interest_hit_rate = loc_interest_hits / total if total > 0 else 0
    loc_location_hit_rate = loc_location_hits / (len(sample_users) * 5) if sample_users else 0

    print(f"Standard approach:")
    print(f"  - Interest hit rate: {std_interest_hit_rate:.4f} ({std_interest_hits}/{total})")
    print(f"  - Location relevance rate: {std_location_hit_rate:.4f} ({std_location_hits}/{len(sample_users)*5})")

    print(f"Location priority approach:")
    print(f"  - Interest hit rate: {loc_interest_hit_rate:.4f} ({loc_interest_hits}/{total})")
    print(f"  - Location relevance rate: {loc_location_hit_rate:.4f} ({loc_location_hits}/{len(sample_users)*5})")

    return {
        'standard': {
            'interest_hit_rate': std_interest_hit_rate,
            'location_hit_rate': std_location_hit_rate
        },
        'location_priority': {
            'interest_hit_rate': loc_interest_hit_rate,
            'location_hit_rate': loc_location_hit_rate
        }
    }

# Generate final recommendations for submission
def generate_final_recommendations(output_path='recommendations.csv', batch_size=1000):
    """Generate recommendations for all test users and save to CSV"""
    # Load test users
    test = pd.read_csv("/content/drive/MyDrive/d2k/test.csv")
    test_users = test['user'].astype(str).unique()

    print(f"Generating recommendations for {len(test_users)} test users...")

    # Create output dataframe
    all_recommendations = []

    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(test_users), batch_size)):
        batch_users = test_users[i:i+batch_size]

        for user in batch_users:
            # Decide which recommendation method to use
            if user in user_locations:
                # Use location priority for users with location data
                recommendations = recommend_events_with_location_priority(user, num_recommendations=10)
            else:
                # Use standard approach for users without location data
                recommendations = recommend_events(user, num_recommendations=10)

            if len(recommendations) > 0:
                # Format for submission
                for _, row in recommendations.iterrows():
                    all_recommendations.append({
                        'user': user,
                        'event': row['event_id']
                    })

        # Checkpoint save to avoid losing progress
        if i % (batch_size * 5) == 0 and i > 0:
            temp_df = pd.DataFrame(all_recommendations)
            temp_df.to_csv(f"{output_path}.checkpoint_{i}", index=False)
            print(f"Checkpoint saved at {i} users")

    # Convert to dataframe and save
    print(f"Generated {len(all_recommendations)} recommendations")
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df.to_csv(output_path, index=False)
    print(f"Recommendations saved to {output_path}")

    return recommendations_df

# Function to run the entire pipeline
def run_recommendation_pipeline():
    """Run the complete recommendation pipeline"""
    print("Starting recommendation pipeline...")

    # Test the recommendation system on a small sample
    print("\n=== Testing Recommendations ===")
    test_location_recommendations()

    # Evaluate recommendation approaches
    print("\n=== Evaluating Recommendation Approaches ===")
    eval_results = compare_recommendation_approaches(num_samples=50)

    # Generate final recommendations
    print("\n=== Generating Final Recommendations ===")
    recommendations = generate_final_recommendations()

    print("\n=== Pipeline Complete ===")
    return recommendations

# Execute the pipeline if running as main script
if __name__ == "__main__":
    run_recommendation_pipeline()