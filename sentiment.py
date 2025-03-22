import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import gc  # For garbage collection
warnings.filterwarnings('ignore')

class EventPlanner:
    def __init__(self, data_directory=None, articles_path=None, clicks_sample_path=None, clicks_folder=None, pickle_path=None):
        """
        Initialize EventPlanner with data directory path or specific file paths

        Parameters:
        data_directory (str): Path to the directory containing all data files
        articles_path (str): Path to articles metadata CSV
        clicks_sample_path (str): Path to clicks sample CSV
        clicks_folder (str): Path to folder containing hourly clicks data
        pickle_path (str): Path to articles embeddings pickle file
        """
        self.data_directory = data_directory
        self.articles_path = articles_path
        self.clicks_sample_path = clicks_sample_path
        self.clicks_folder = clicks_folder
        self.pickle_path = pickle_path

        self.articles_metadata = None
        self.clicks_sample = None
        self.clicks_data = None
        self.user_preferences = None
        self.location_clusters = None
        self.article_similarities = {}
        self.user_vectors = None
        self.popular_categories = None
        self.loaded_model = None

    def load_data(self, sample_size=None):
        """
        Load all relevant datasets
        """
        try:
            metadata_path = self.articles_path if self.articles_path else os.path.join(self.data_directory, 'articles_metadata.csv')
            metadata_cols = ['article_id', 'category_id', 'created_at_ts', 'words_count']
            self.articles_metadata = pd.read_csv(metadata_path, usecols=metadata_cols)

            self.articles_metadata['created_at'] = pd.to_datetime(self.articles_metadata['created_at_ts'], unit='ms')
            self.articles_metadata.drop('created_at_ts', axis=1, inplace=True)
            self.articles_metadata['article_id'] = self.articles_metadata['article_id'].astype('int32')
            self.articles_metadata['category_id'] = self.articles_metadata['category_id'].astype('int16')

            clicks_sample_path = self.clicks_sample_path if self.clicks_sample_path else os.path.join(self.data_directory, 'clicks_sample.csv')
            clicks_cols = ['user_id', 'click_article_id', 'session_start', 'click_timestamp',
                         'click_country', 'click_region', 'click_deviceGroup',
                         'click_environment', 'session_size']

            self.clicks_sample = pd.read_csv(clicks_sample_path, usecols=clicks_cols,
                                           nrows=sample_size if sample_size else None)

            self.clicks_sample['session_start_time'] = pd.to_datetime(self.clicks_sample['session_start'], unit='ms')
            self.clicks_sample['click_time'] = pd.to_datetime(self.clicks_sample['click_timestamp'], unit='ms')
            self.clicks_sample.drop(['session_start', 'click_timestamp'], axis=1, inplace=True)
            self.clicks_sample['user_id'] = self.clicks_sample['user_id'].astype('int32')
            self.clicks_sample['click_article_id'] = self.clicks_sample['click_article_id'].astype('int32')
            self.clicks_sample['session_size'] = self.clicks_sample['session_size'].astype('int16')

            try:
                pickle_path = self.pickle_path if self.pickle_path else os.path.join(self.data_directory, 'articles_embeddings.pickle')
                with open(pickle_path, 'rb') as f:
                    self.loaded_model = pickle.load(f)
                print("Loaded pickle model successfully")
            except Exception as e:
                print(f"Note: Could not load pickle file: {e}")

            print("Data loaded successfully!")
            gc.collect()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def load_clicks_data(self, max_files=20, sample_rows=10000):
        """
        Load and combine the hourly clicks data
        """
        clicks_dir = self.clicks_folder if self.clicks_folder else os.path.join(self.data_directory, 'clicks')

        if not os.path.exists(clicks_dir):
            print(f"Warning: Clicks directory not found at {clicks_dir}")
            return

        all_clicks = []
        files_loaded = 0
        click_cols = ['user_id', 'click_article_id', 'click_timestamp',
                     'click_country', 'click_region']

        for i in range(356):
            if files_loaded >= max_files:
                break

            file_path = os.path.join(clicks_dir, f'clicks_hour_{i:03d}.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, usecols=click_cols, nrows=sample_rows)
                    df['user_id'] = df['user_id'].astype('int32')
                    df['click_article_id'] = df['click_article_id'].astype('int32')
                    all_clicks.append(df)
                    files_loaded += 1
                    if files_loaded % 5 == 0:
                        gc.collect()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        if all_clicks:
            batch_size = 5
            final_clicks = []
            for i in range(0, len(all_clicks), batch_size):
                batch = pd.concat(all_clicks[i:i+batch_size], ignore_index=True)
                final_clicks.append(batch)
                all_clicks[i:i+batch_size] = [None] * min(batch_size, len(all_clicks) - i)
                gc.collect()

            self.clicks_data = pd.concat(final_clicks, ignore_index=True)
            self.clicks_data['click_time'] = pd.to_datetime(self.clicks_data['click_timestamp'], unit='ms')
            self.clicks_data.drop('click_timestamp', axis=1, inplace=True)
            print(f"Loaded {files_loaded} clicks data files (sampled {sample_rows} rows each)")
            gc.collect()
        else:
            print("No clicks data files were loaded successfully")

    def analyze_user_preferences(self, max_users=10000):
        """
        Analyze user preferences for content and locations
        """
        if self.clicks_sample is None:
            print("Please load data first using load_data()")
            return

        unique_users = self.clicks_sample['user_id'].unique()
        if len(unique_users) > max_users:
            selected_users = np.random.choice(unique_users, max_users, replace=False)
            sample_clicks = self.clicks_sample[self.clicks_sample['user_id'].isin(selected_users)]
        else:
            sample_clicks = self.clicks_sample

        if self.articles_metadata is not None:
            clicks_with_categories = pd.merge(
                sample_clicks[['user_id', 'click_article_id']],
                self.articles_metadata[['article_id', 'category_id']],
                left_on='click_article_id',
                right_on='article_id',
                how='left'
            )

            batch_size = 50000
            user_category_list = []
            for i in range(0, len(clicks_with_categories), batch_size):
                batch = clicks_with_categories.iloc[i:i+batch_size]
                user_category_batch = batch.groupby(['user_id', 'category_id']).size().reset_index(name='count')
                user_category_list.append(user_category_batch)
                del batch
                gc.collect()

            user_category_preferences = pd.concat(user_category_list, ignore_index=True)
            user_category_preferences = user_category_preferences.groupby(['user_id', 'category_id'])['count'].sum().reset_index()
            pivot_df = user_category_preferences.pivot(index='user_id', columns='category_id', values='count').fillna(0)
            row_sums = pivot_df.sum(axis=1)
            self.user_preferences = pivot_df.div(row_sums, axis=0)
            self.popular_categories = clicks_with_categories['category_id'].value_counts().head(10).to_dict()
            del clicks_with_categories, user_category_list, user_category_preferences, pivot_df
            gc.collect()

        if 'click_country' in sample_clicks.columns and 'click_region' in sample_clicks.columns:
            location_data = sample_clicks.groupby(['user_id', 'click_country', 'click_region']).size().reset_index(name='visits')
            self.location_clusters = location_data.sort_values('visits', ascending=False).head(1000)

        if self.articles_metadata is not None and len(self.articles_metadata) > 0:
            max_articles = 1000
            sampled_articles = self.articles_metadata.sample(max_articles, random_state=42) if len(self.articles_metadata) > max_articles else self.articles_metadata
            content_features = pd.get_dummies(sampled_articles['category_id'], prefix='category')
            if 'words_count' in sampled_articles.columns:
                content_features['words_count'] = sampled_articles['words_count'] / sampled_articles['words_count'].max()

            article_ids = sampled_articles['article_id'].values
            batch_size = 200
            for i in range(0, len(article_ids), batch_size):
                batch_ids = article_ids[i:i+batch_size]
                batch_features = content_features.iloc[i:i+batch_size]
                similarity_batch = cosine_similarity(batch_features, content_features)
                for idx, article_id in enumerate(batch_ids):
                    similarities = similarity_batch[idx]
                    if i + idx < len(similarities):
                        similarities[i + idx] = -1
                    top_indices = np.argsort(similarities)[-10:]
                    self.article_similarities[article_id] = {
                        article_ids[j]: similarities[j]
                        for j in top_indices if similarities[j] > 0
                    }
                del batch_features, similarity_batch
                gc.collect()

            if len(sample_clicks) > 0:
                self.user_vectors = self.user_preferences

        print("User preference analysis completed with memory optimizations")
        gc.collect()

    def recommend_events(self, user_id=None, n_recommendations=5):
        """
        Recommend events/articles Noc
        """
        if self.articles_metadata is None:
            print("Please load data first using load_data()")
            return None

        if user_id is not None and self.user_preferences is not None and user_id in self.user_preferences.index:
            user_cats = self.user_preferences.loc[user_id].sort_values(ascending=False).index[:5]
            cat_articles = self.articles_metadata[self.articles_metadata['category_id'].isin(user_cats)]
            if self.clicks_sample is not None:
                user_history = set(self.clicks_sample[self.clicks_sample['user_id'] == user_id]['click_article_id'])
                cat_articles = cat_articles[~cat_articles['article_id'].isin(user_history)]
            if len(cat_articles) >= n_recommendations:
                return cat_articles.head(n_recommendations)

        if self.clicks_sample is not None:
            popular_articles = self.clicks_sample['click_article_id'].value_counts().head(n_recommendations).index
            general_recommendations = self.articles_metadata[self.articles_metadata['article_id'].isin(popular_articles)]
            return general_recommendations

        return self.articles_metadata.sort_values('created_at', ascending=False).head(n_recommendations)

    def predict_event_popularity(self, timeframe_days=7, max_articles=100):
        """
        Predict which events/articles will be popular
        """
        if self.clicks_sample is None or self.articles_metadata is None:
            print("Please load data first using load_data()")
            return None

        clicks_to_analyze = self.clicks_sample.sample(100000, random_state=42) if len(self.clicks_sample) > 100000 else self.clicks_sample

        if len(clicks_to_analyze) > 0:
            latest_date = clicks_to_analyze['click_time'].max()
            clicks_to_analyze['days_ago'] = (latest_date - clicks_to_analyze['click_time']).dt.days + 1
            clicks_to_analyze['recency_weight'] = 1 / clicks_to_analyze['days_ago']
            weighted_popularity = clicks_to_analyze[['click_article_id', 'recency_weight']].groupby('click_article_id')['recency_weight'].sum().reset_index()
            weighted_popularity = weighted_popularity.rename(columns={'recency_weight': 'popularity_score'})
            weighted_popularity = weighted_popularity.sort_values('popularity_score', ascending=False).head(max_articles)

            popular_predictions = pd.merge(
                weighted_popularity,
                self.articles_metadata[['article_id', 'category_id']],
                left_on='click_article_id',
                right_on='article_id',
                how='left'
            )

            max_score = popular_predictions['popularity_score'].max()
            popular_predictions['prediction_confidence'] = popular_predictions['popularity_score'] / max_score if max_score > 0 else 0
            return popular_predictions
        return None

    def get_location_preferences(self, user_id=None, max_locations=20):
        """
        Get location preferences
        """
        if self.clicks_sample is None:
            print("Please load data first using load_data()")
            return None

        if 'click_country' not in self.clicks_sample.columns or 'click_region' not in self.clicks_sample.columns:
            print("Location data not available in the dataset")
            return None

        if user_id is not None:
            user_locations = self.clicks_sample[self.clicks_sample['user_id'] == user_id][['click_country', 'click_region']]
            if len(user_locations) > 0:
                location_prefs = user_locations.groupby(['click_country', 'click_region']).size().reset_index(name='frequency')
                return location_prefs.sort_values('frequency', ascending=False).head(max_locations)

        sample_size = min(100000, len(self.clicks_sample))
        location_sample = self.clicks_sample.sample(sample_size, random_state=42)[['click_country', 'click_region']] if len(self.clicks_sample) > sample_size else self.clicks_sample[['click_country', 'click_region']]
        general_locations = location_sample.groupby(['click_country', 'click_region']).size().reset_index(name='frequency')
        return general_locations.sort_values('frequency', ascending=False).head(max_locations)

    def generate_insights(self, sample_size=50000):
        """
        Generate insights from the data
        """
        insights = {}
        if self.clicks_sample is None or self.articles_metadata is None:
            print("Please load data first using load_data()")
            return insights

        clicks_sample = self.clicks_sample.sample(sample_size, random_state=42) if len(self.clicks_sample) > sample_size else self.clicks_sample

        if 'click_time' in clicks_sample.columns:
            hours = clicks_sample['click_time'].dt.hour.values
            days = clicks_sample['click_time'].dt.dayofweek.values
            hour_counts = np.bincount(hours, minlength=24)
            day_counts = np.bincount(days, minlength=7)
            insights['peak_hours'] = np.argsort(hour_counts)[-3:].tolist()
            peak_day_indices = np.argsort(day_counts)[-3:].tolist()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            insights['peak_days'] = [day_names[day] for day in peak_day_indices]

        if 'click_deviceGroup' in clicks_sample.columns:
            device_counts = clicks_sample['click_deviceGroup'].value_counts()
            insights['device_preference'] = device_counts.index[0] if len(device_counts) > 0 else None

        if 'click_environment' in clicks_sample.columns:
            env_counts = clicks_sample['click_environment'].value_counts()
            insights['environment_preference'] = env_counts.index[0] if len(env_counts) > 0 else None

        if 'session_size' in clicks_sample.columns:
            insights['avg_session_size'] = clicks_sample['session_size'].mean()
            insights['median_session_size'] = clicks_sample['session_size'].median()

        if self.articles_metadata is not None and 'category_id' in self.articles_metadata.columns:
            sample_clicks = clicks_sample[['click_article_id']].copy()
            category_data = pd.merge(
                sample_clicks,
                self.articles_metadata[['article_id', 'category_id']],
                left_on='click_article_id',
                right_on='article_id',
                how='left'
            )
            category_popularity = category_data['category_id'].value_counts()
            insights['popular_categories'] = category_popularity.nlargest(3).index.tolist()
            del sample_clicks, category_data
            gc.collect()

        if 'user_id' in clicks_sample.columns:
            user_activity = clicks_sample['user_id'].value_counts()
            insights['active_users_count'] = (user_activity > 5).sum()
            insights['power_users_count'] = (user_activity > 20).sum()
            if 'click_time' in clicks_sample.columns:
                top_users = user_activity.nlargest(10000).index
                user_times = [(user, clicks_sample[clicks_sample['user_id'] == user]['click_time'].min(),
                              clicks_sample[clicks_sample['user_id'] == user]['click_time'].max())
                             for user in top_users if len(clicks_sample[clicks_sample['user_id'] == user]) > 0]
                if user_times:
                    user_first_last = pd.DataFrame(user_times, columns=['user_id', 'first', 'last'])
                    user_first_last['days_active'] = (user_first_last['last'] - user_first_last['first']).dt.days
                    insights['avg_user_active_days'] = user_first_last['days_active'].mean()
                    insights['returning_users_pct'] = (user_first_last['days_active'] > 0).mean() * 100
                    del user_first_last

        del clicks_sample
        gc.collect()
        return insights

    def visualize_insights(self, save_path=None, sample_size=50000):
        """
        Create visualizations for key insights
        """
        if self.clicks_sample is None:
            print("Please load data first using load_data()")
            return

        clicks_sample = self.clicks_sample.sample(sample_size, random_state=42) if len(self.clicks_sample) > sample_size else self.clicks_sample

        plt.figure(figsize=(15, 10))
        if 'click_time' in clicks_sample.columns:
            hours = clicks_sample['click_time'].dt.hour.values
            hour_counts = np.bincount(hours, minlength=24)
            plt.subplot(2, 2, 1)
            plt.bar(range(24), hour_counts)
            plt.title('Hourly Activity Pattern')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Clicks')
            plt.xticks(range(0, 24, 3))

            days = clicks_sample['click_time'].dt.dayofweek.values
            day_counts = np.bincount(days, minlength=7)
            plt.subplot(2, 2, 2)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            plt.bar(day_names, day_counts)
            plt.title('Daily Activity Pattern')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Clicks')

        if 'click_country' in clicks_sample.columns:
            country_counts = clicks_sample['click_country'].value_counts().head(10)
            plt.subplot(2, 2, 3)
            plt.bar(country_counts.index, country_counts.values)
            plt.title('Top 10 Countries by Activity')
            plt.xlabel('Country ID')
            plt.ylabel('Number of Clicks')
            plt.xticks(rotation=45)

        if 'click_deviceGroup' in clicks_sample.columns:
            device_counts = clicks_sample['click_deviceGroup'].value_counts()
            plt.subplot(2, 2, 4)
            plt.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%')
            plt.title('Device Distribution')

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'insights_visualization.png'), dpi=100)
            plt.close()
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            plt.close()

        del clicks_sample
        gc.collect()

    def run_full_analysis(self, user_id=None, sample_size=50000):
        """
        Run a complete analysis with memory optimizations

        Parameters:
        user_id (int): Optional user ID for personalized analysis
        sample_size (int): Maximum number of rows to sample for large datasets

        Returns:
        dict: Complete analysis report
        """
        report = {}

        # 1. Load all data with sampling
        load_success = self.load_data(sample_size=sample_size)
        if not load_success:
            return {"error": "Failed to load data"}

        # 2. Load clicks data
        self.load_clicks_data()

        # 3. Analyze user preferences with limits
        self.analyze_user_preferences(max_users=5000)

        # 4. Get recommendations
        if user_id:
            recommendations = self.recommend_events(user_id)
            if recommendations is not None:
                report['personalized_recommendations'] = recommendations[['article_id', 'category_id']].head(5).to_dict('records')
        else:
            general_recommendations = self.recommend_events()
            if general_recommendations is not None:
                report['general_recommendations'] = general_recommendations[['article_id', 'category_id']].head(5).to_dict('records')

        # 5. Predict popular events
        popularity_predictions = self.predict_event_popularity(max_articles=50)
        if popularity_predictions is not None:
            report['predicted_popular_events'] = popularity_predictions[['click_article_id', 'category_id', 'popularity_score']].head(5).to_dict('records')

        # 6. Get location preferences
        if user_id:
            location_prefs = self.get_location_preferences(user_id, max_locations=10)
            if location_prefs is not None:
                report['location_preferences'] = location_prefs.head(5).to_dict('records')
        else:
            general_location_prefs = self.get_location_preferences(max_locations=10)
            if general_location_prefs is not None:
                report['general_location_preferences'] = general_location_prefs.head(5).to_dict('records')

        # 7. Generate insights
        insights = self.generate_insights(sample_size=sample_size)
        report['insights'] = insights

        # 8. Create visualizations
        try:
            save_path = os.path.join(self.data_directory, 'visualizations') if self.data_directory else '/content/drive/MyDrive/D2K_2/visualizations'
            self.visualize_insights(save_path=save_path, sample_size=sample_size)
            report['visualization_created'] = True
        except Exception as e:
            report['visualization_error'] = str(e)

        # 9. Free memory before returning
        gc.collect()

        return report

# Example usage
if __name__ == "__main__":

    SAMPLE_SIZE = 50000

    # Initialize with specific paths for Colab
    planner = EventPlanner(
        articles_path='/content/drive/MyDrive/D2K_2/articles_metadata.csv',
        clicks_sample_path='/content/drive/MyDrive/D2K_2/clicks_sample.csv',
        clicks_folder='/content/drive/MyDrive/D2K_2/clicks',
        pickle_path='/content/drive/MyDrive/D2K_2/articles_embeddings.pickle'
    )

    # Run full analysis
    user_id = 1506825423271737  # Example user ID from your sample data
    report = planner.run_full_analysis(user_id=user_id, sample_size=SAMPLE_SIZE)

    # Print results
    print("\nAnalysis Report:")
    for key, value in report.items():
        print(f"{key}: {value}")