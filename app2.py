import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Organizer Intelligence Dashboard",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1976D2;
        margin-top: 1rem;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2196F3;
    }
    .metric-container {
        background-color: #f5f5f5;
        border-radius: 7px;
        padding: 10px;
        text-align: center;
    }
    .highlight {
        color: #FF5722;
        font-weight: bold;
    }
    .info-text {
        font-size: 1rem;
    }
    .metric-container p{
        color:black
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the saved CSV file
        df = pd.read_csv("event_data.csv")
        
        # Convert date columns to datetime
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'])
            
        # Load the JSON results file for additional insights
        with open("organizer_intelligence_results.json", "r") as f:
            results = json.load(f)
            
        return df, results
    except:
        # Generate sample data if files are not found
        st.warning("Could not load saved files. Generating sample data for demonstration.")
        # Create date range
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="W")
        artists = ["Artist A", "Artist B", "Artist C", "Artist D"]
        categories = ["Concert", "Festival", "Workshop"]
        
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        data = {
            "event_date": np.random.choice(dates, 100),
            "event_category": np.random.choice(categories, 100),
            "tickets_sold": np.random.randint(50, 1000, 100),
            "ticket_price": np.random.uniform(20, 200, 100),
            "days_to_event": np.random.randint(1, 90, 100),
            "x_text": [f"Event {i} feedback: {'great' if i % 2 == 0 else 'poor'}" for i in range(100)],
            "sentiment_score": [0.6 if i % 2 == 0 else -0.4 for i in range(100)],
            "is_anomaly": np.random.choice([-1, 1], 100, p=[0.1, 0.9]),
            "anomaly_score": np.random.normal(-0.5, 0.2, 100)
        }
        
        df = pd.DataFrame(data)
        
        # Add derived columns
        df['month'] = df['event_date'].dt.month
        df['day_of_week'] = df['event_date'].dt.dayofweek
        df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['revenue'] = df['tickets_sold'] * df['ticket_price']
        df['sentiment_category'] = pd.cut(
            df['sentiment_score'],
            bins=[-1, -0.5, 0.0, 0.5, 1],
            labels=['very negative', 'negative', 'neutral', 'positive']
        )
        
        # Create sample results
        results = {
            "summary": {
                "events_analyzed": len(df),
                "date_range": f"{df['event_date'].min().date()} to {df['event_date'].max().date()}",
                "total_tickets_sold": int(df['tickets_sold'].sum()),
                "avg_tickets_per_event": float(df['tickets_sold'].mean()),
                "seasonal_insights": {
                    "best_month": df.groupby('month')['tickets_sold'].mean().idxmax(),
                    "worst_month": df.groupby('month')['tickets_sold'].mean().idxmin()
                },
                "top_events": df.nlargest(5, 'tickets_sold')[['event_date', 'tickets_sold', 'ticket_price']].to_dict(),
                "avg_sentiment": float(df['sentiment_score'].mean()),
                "anomalies_detected": int((df['is_anomaly'] == -1).sum())
            },
            "pricing_strategies": {
                category: {
                    "revenue_maximizing": {
                        "price": float(df[df['event_category'] == category]['ticket_price'].mean() * 1.2),
                        "estimated_attendance": float(df[df['event_category'] == category]['tickets_sold'].mean() * 0.9),
                        "estimated_revenue": float(df[df['event_category'] == category]['tickets_sold'].mean() * 0.9 * 
                                             df[df['event_category'] == category]['ticket_price'].mean() * 1.2)
                    },
                    "attendance_maximizing": {
                        "price": float(df[df['event_category'] == category]['ticket_price'].mean() * 0.8),
                        "estimated_attendance": float(df[df['event_category'] == category]['tickets_sold'].mean() * 1.2),
                        "estimated_revenue": float(df[df['event_category'] == category]['tickets_sold'].mean() * 1.2 * 
                                             df[df['event_category'] == category]['ticket_price'].mean() * 0.8)
                    }
                } for category in categories
            },
            "sentiment": {
                "average": float(df['sentiment_score'].mean()),
                "distribution": df['sentiment_category'].value_counts().to_dict()
            },
            "anomalies": {
                "total": int((df['is_anomaly'] == -1).sum()),
                "percentage": float((df['is_anomaly'] == -1).sum() / len(df) * 100)
            }
        }
            
        return df, results

# Main function to build the app
def main():
    # App title and header
    st.markdown("<h1 class='main-header'>🎭 Organizer Intelligence Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    df, results = load_data()
    
    # Summary metrics at the top
    summary = results.get("summary", {})
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Events Analyzed", f"{summary.get('events_analyzed', 0):,}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Total Tickets Sold", f"{summary.get('total_tickets_sold', 0):,}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Avg. Tickets/Event", f"{summary.get('avg_tickets_per_event', 0):.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Anomalies Detected", f"{summary.get('anomalies_detected', 0)} ({results.get('anomalies', {}).get('percentage', 0):.1f}%)")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Analysis", "💰 Pricing Strategy", "😊 Sentiment Analysis", "⚠️ Anomaly Detection"])
    
    with tab1:
        st.markdown("<h2 class='section-header'>Sales & Seasonal Trends</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly sales trend
            monthly_sales = df.groupby(df['event_date'].dt.to_period('M')).agg({
                'tickets_sold': 'sum',
                'revenue': 'sum'
            }).reset_index()
            monthly_sales['event_date'] = monthly_sales['event_date'].dt.to_timestamp()
            
            fig = px.line(monthly_sales, x='event_date', y=['tickets_sold', 'revenue'], 
                         title='Monthly Ticket Sales & Revenue',
                         labels={'event_date': 'Month', 'value': 'Amount', 'variable': 'Metric'},
                         line_shape='linear')
            fig.update_layout(hovermode='x unified', legend_title_text='')
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight best and worst months
            best_month = summary.get('seasonal_insights', {}).get('best_month', 0)
            worst_month = summary.get('seasonal_insights', {}).get('worst_month', 0)
            
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            
            st.markdown(f"""
            <div class='info-text'>
                <span class='highlight'>Best Month:</span> {month_names.get(best_month, 'N/A')} |
                <span class='highlight'>Worst Month:</span> {month_names.get(worst_month, 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Day of week analysis
            dow_data = df.groupby('day_of_week').agg({
                'tickets_sold': 'mean',
                'revenue': 'mean'
            }).reset_index()
            
            dow_names = {
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            }
            dow_data['day_name'] = dow_data['day_of_week'].map(dow_names)
            
            fig = px.bar(dow_data, x='day_name', y='tickets_sold', 
                        title='Average Ticket Sales by Day of Week',
                        labels={'day_name': 'Day', 'tickets_sold': 'Avg. Tickets Sold'},
                        color='tickets_sold',
                        color_continuous_scale='Blues')
            
            fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': list(dow_names.values())})
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekend vs Weekday comparison
            weekend_data = df.groupby('weekend').agg({
                'tickets_sold': 'mean',
                'ticket_price': 'mean',
                'revenue': 'mean'
            }).reset_index()
            weekend_data['weekend'] = weekend_data['weekend'].map({0: 'Weekday', 1: 'Weekend'})
            
            fig = px.bar(weekend_data, x='weekend', y=['tickets_sold', 'revenue'], 
                        barmode='group',
                        title='Weekend vs Weekday Performance',
                        labels={'weekend': '', 'value': 'Amount', 'variable': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Event category performance
        st.markdown("<h3 class='sub-header'>Event Category Performance</h3>", unsafe_allow_html=True)
        category_data = df.groupby('event_category').agg({
            'tickets_sold': ['mean', 'sum'],
            'ticket_price': 'mean',
            'revenue': ['mean', 'sum'],
            'sentiment_score': 'mean'
        }).reset_index()
        
        category_data.columns = ['event_category', 'avg_tickets', 'total_tickets', 
                               'avg_price', 'avg_revenue', 'total_revenue', 'avg_sentiment']
        
        fig = px.bar(category_data, x='event_category', y=['avg_tickets', 'avg_revenue', 'avg_sentiment'],
                   title='Performance by Event Category',
                   barmode='group',
                   labels={'event_category': 'Category', 'value': 'Value', 'variable': 'Metric'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top 5 events
        st.markdown("<h3 class='sub-header'>Top Performing Events</h3>", unsafe_allow_html=True)
        top_events = df.nlargest(5, 'tickets_sold')[['event_date', 'event_category', 'tickets_sold', 'ticket_price']]
        top_events['revenue'] = top_events['tickets_sold'] * top_events['ticket_price']
        top_events['event_date'] = top_events['event_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(top_events, use_container_width=True)

    with tab2:
        st.markdown("<h2 class='section-header'>Pricing Strategy Optimization</h2>", unsafe_allow_html=True)
        
        # Get pricing strategies from results
        pricing_strategies = results.get("pricing_strategies", {})
        
        if pricing_strategies:
            # Create dropdown to select event category
            categories = list(pricing_strategies.keys())
            selected_category = st.selectbox("Select Event Category", categories)
            
            if selected_category in pricing_strategies:
                strategy = pricing_strategies[selected_category]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Revenue maximizing strategy
                    st.markdown("<h3 class='sub-header'>Revenue Maximizing Strategy</h3>", unsafe_allow_html=True)
                    
                    rev_max = strategy.get("revenue_maximizing", {})
                    
                    st.markdown(f"""
                    <div class='metric-container'>
                        <p><span class='highlight'>Optimal Price:</span> ${rev_max.get('price', 0):.2f}</p>
                        <p><span class='highlight'>Est. Attendance:</span> {rev_max.get('estimated_attendance', 0):.0f}</p>
                        <p><span class='highlight'>Est. Revenue:</span> ${rev_max.get('estimated_revenue', 0):,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Attendance maximizing strategy
                    st.markdown("<h3 class='sub-header'>Attendance Maximizing Strategy</h3>", unsafe_allow_html=True)
                    
                    att_max = strategy.get("attendance_maximizing", {})
                    
                    st.markdown(f"""
                    <div class='metric-container'>
                        <p><span class='highlight'>Optimal Price:</span> ${att_max.get('price', 0):.2f}</p>
                        <p><span class='highlight'>Est. Attendance:</span> {att_max.get('estimated_attendance', 0):.0f}</p>
                        <p><span class='highlight'>Est. Revenue:</span> ${att_max.get('estimated_revenue', 0):,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Strategy comparison
                st.markdown("<h3 class='sub-header'>Strategy Comparison</h3>", unsafe_allow_html=True)
                
                # Create comparison data for visualization
                comparison_data = {
                    "Strategy": ["Revenue Maximizing", "Attendance Maximizing"],
                    "Price": [rev_max.get('price', 0), att_max.get('price', 0)],
                    "Attendance": [rev_max.get('estimated_attendance', 0), att_max.get('estimated_attendance', 0)],
                    "Revenue": [rev_max.get('estimated_revenue', 0), att_max.get('estimated_revenue', 0)]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create comparison chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comparison_df['Strategy'],
                    y=comparison_df['Price'],
                    name='Price',
                    marker_color='rgb(55, 83, 109)',
                    text=[f'${x:.2f}' for x in comparison_df['Price']],
                    textposition='auto',
                ))
                
                fig.add_trace(go.Bar(
                    x=comparison_df['Strategy'],
                    y=comparison_df['Attendance'],
                    name='Attendance',
                    marker_color='rgb(26, 118, 255)',
                    text=[f'{x:.0f}' for x in comparison_df['Attendance']],
                    textposition='auto',
                ))
                
                fig.add_trace(go.Bar(
                    x=comparison_df['Strategy'],
                    y=comparison_df['Revenue'],
                    name='Revenue',
                    marker_color='rgb(33, 150, 83)',
                    text=[f'${x:,.0f}' for x in comparison_df['Revenue']],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title='Strategy Comparison',
                    xaxis_title='Strategy',
                    yaxis_title='Value',
                    barmode='group',
                    legend_title='Metric',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price elasticity analysis
                st.markdown("<h3 class='sub-header'>Price Elasticity Analysis</h3>", unsafe_allow_html=True)
                
                # Create price range simulation
                current_avg_price = df[df['event_category'] == selected_category]['ticket_price'].mean()
                current_avg_attendance = df[df['event_category'] == selected_category]['tickets_sold'].mean()
                
                price_range = np.linspace(current_avg_price * 0.5, current_avg_price * 1.5, 20)
                elasticity = -0.7  # Assumed price elasticity
                
                attendance_impact = []
                revenue_impact = []
                
                for price in price_range:
                    price_diff_pct = (price - current_avg_price) / current_avg_price
                    att_impact = 1 + (price_diff_pct * elasticity)
                    est_attendance = current_avg_attendance * att_impact
                    attendance_impact.append(est_attendance)
                    revenue_impact.append(price * est_attendance)
                
                elasticity_df = pd.DataFrame({
                    'Price': price_range,
                    'Est. Attendance': attendance_impact,
                    'Est. Revenue': revenue_impact
                })
                
                fig = px.line(elasticity_df, x='Price', y=['Est. Attendance', 'Est. Revenue'],
                            title='Price Elasticity Simulation',
                            labels={'Price': 'Ticket Price ($)', 'value': 'Value', 'variable': 'Metric'},
                            line_shape='linear')
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class='info-text'>
                This chart shows how ticket price affects both attendance and revenue. 
                The optimal price point depends on your goals:
                <ul>
                    <li>Maximize revenue: Set price where the revenue curve peaks</li>
                    <li>Maximize attendance: Set price where the attendance curve is highest</li>
                    <li>Balance both: Find a middle ground between the two curves</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No pricing strategy data available.")

    with tab3:
        st.markdown("<h2 class='section-header'>Sentiment Analysis</h2>", unsafe_allow_html=True)
        
        # Get sentiment data
        sentiment_data = results.get("sentiment", {})
        
        if sentiment_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall sentiment distribution
                st.markdown("<h3 class='sub-header'>Sentiment Distribution</h3>", unsafe_allow_html=True)
                
                distribution = sentiment_data.get("distribution", {})
                
                if distribution:
                    sentiment_dist_df = pd.DataFrame({
                        'Sentiment': list(distribution.keys()),
                        'Count': list(distribution.values())
                    })
                    
                    # Sort by sentiment in proper order
                    sentiment_order = ['very negative', 'negative', 'neutral', 'positive']
                    sentiment_dist_df['Sentiment'] = pd.Categorical(
                        sentiment_dist_df['Sentiment'], 
                        categories=sentiment_order, 
                        ordered=True
                    )
                    sentiment_dist_df = sentiment_dist_df.sort_values('Sentiment')
                    
                    colors = ['#d7191c', '#fdae61', '#ffffbf', '#1a9641']
                    
                    fig = px.pie(sentiment_dist_df, names='Sentiment', values='Count',
                                title='Sentiment Distribution',
                                color='Sentiment',
                                color_discrete_map=dict(zip(sentiment_order, colors)))
                    
                    fig.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    avg_sentiment = sentiment_data.get("average", 0)
                    sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
                    
                    st.markdown(f"""
                    <div class='info-text'>
                        <p><span class='highlight'>Average Sentiment Score:</span> {avg_sentiment:.2f} ({sentiment_label})</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Sentiment by category
                st.markdown("<h3 class='sub-header'>Sentiment by Event Category</h3>", unsafe_allow_html=True)
                
                category_sentiment = df.groupby('event_category')['sentiment_score'].mean().reset_index()
                
                fig = px.bar(category_sentiment, x='event_category', y='sentiment_score',
                            title='Average Sentiment by Event Category',
                            labels={'event_category': 'Category', 'sentiment_score': 'Avg. Sentiment'},
                            color='sentiment_score',
                            color_continuous_scale='RdYlGn')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment trend over time
            st.markdown("<h3 class='sub-header'>Sentiment Trend Over Time</h3>", unsafe_allow_html=True)
            
            sentiment_trend = df.sort_values('event_date').set_index('event_date')['sentiment_score'].rolling(window=5).mean()
            sentiment_trend = sentiment_trend.reset_index()
            
            fig = px.line(sentiment_trend, x='event_date', y='sentiment_score',
                        title='Sentiment Trend Over Time (5-Event Rolling Average)',
                        labels={'event_date': 'Date', 'sentiment_score': 'Sentiment Score'},
                        line_shape='linear')
            
            # Add a zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sample feedback
            st.markdown("<h3 class='sub-header'>Sample Feedback</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<p class='highlight'>Most Positive Feedback</p>", unsafe_allow_html=True)
                positive_feedback = df.nlargest(3, 'sentiment_score')[['x_text', 'sentiment_score']]
                for i, (_, row) in enumerate(positive_feedback.iterrows()):
                    st.markdown(f"""
                    <div class='metric-container' style='margin-bottom: 10px;'>
                        <p>{row['x_text']}</p>
                        <p><i>Sentiment Score: {row['sentiment_score']:.2f}</i></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("<p class='highlight'>Most Negative Feedback</p>", unsafe_allow_html=True)
                negative_feedback = df.nsmallest(3, 'sentiment_score')[['x_text', 'sentiment_score']]
                for i, (_, row) in enumerate(negative_feedback.iterrows()):
                    st.markdown(f"""
                    <div class='metric-container' style='margin-bottom: 10px;'>
                        <p>{row['x_text']}</p>
                        <p><i>Sentiment Score: {row['sentiment_score']:.2f}</i></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No sentiment analysis data available.")

    with tab4:
        st.markdown("<h2 class='section-header'>Anomaly Detection</h2>", unsafe_allow_html=True)
        
        # Get anomaly data
        anomaly_data = results.get("anomalies", {})
        
        if 'is_anomaly' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Anomaly statistics
                st.markdown("<h3 class='sub-header'>Anomaly Statistics</h3>", unsafe_allow_html=True)
                
                total_anomalies = anomaly_data.get("total", (df['is_anomaly'] == -1).sum())
                percent_anomalies = anomaly_data.get("percentage", (df['is_anomaly'] == -1).sum() / len(df) * 100)
                
                st.markdown(f"""
                <div class='metric-container'>
                    <p><span class='highlight'>Total Anomalies:</span> {total_anomalies}</p>
                    <p><span class='highlight'>Percentage of Events:</span> {percent_anomalies:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Anomaly distribution by category
                anomaly_by_category = df.groupby(['event_category', 'is_anomaly']).size().unstack().fillna(0)
                if -1 in anomaly_by_category.columns:
                    anomaly_by_category['anomaly_percent'] = anomaly_by_category[-1] / anomaly_by_category.sum(axis=1) * 100
                    
                    fig = px.bar(anomaly_by_category.reset_index(), x='event_category', y='anomaly_percent',
                                title='Anomaly Percentage by Category',
                                labels={'event_category': 'Category', 'anomaly_percent': '% Anomalies'},
                                color='anomaly_percent',
                                color_continuous_scale='Reds')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Anomaly distribution over time
                st.markdown("<h3 class='sub-header'>Anomalies Over Time</h3>", unsafe_allow_html=True)
                
                anomaly_by_time = df.groupby(pd.Grouper(key='event_date', freq='M'))['is_anomaly'].apply(
                    lambda x: (x == -1).sum() / len(x) * 100
                ).reset_index()
                anomaly_by_time.columns = ['month', 'anomaly_percent']
                
                fig = px.line(anomaly_by_time, x='month', y='anomaly_percent',
                            title='Anomaly Percentage Over Time',
                            labels={'month': 'Month', 'anomaly_percent': '% Anomalies'},
                            line_shape='linear')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            st.markdown("<h3 class='sub-header'>Anomaly Details</h3>", unsafe_allow_html=True)
            
            anomalies_df = df[df['is_anomaly'] == -1].copy()
            
            if len(anomalies_df) > 0:
                # Categorize anomalies by their characteristics
                anomalies_df['high_sales'] = anomalies_df['tickets_sold'] > df['tickets_sold'].quantile(0.9)
                anomalies_df['low_sales'] = anomalies_df['tickets_sold'] < df['tickets_sold'].quantile(0.1)
                anomalies_df['high_price'] = anomalies_df['ticket_price'] > df['ticket_price'].quantile(0.9)
                anomalies_df['low_price'] = anomalies_df['ticket_price'] < df['ticket_price'].quantile(0.1)
                anomalies_df['unusual_sentiment'] = (anomalies_df['sentiment_score'] > 0.8) | (anomalies_df['sentiment_score'] < -0.5)
                
                # Display summary of anomaly types
                anomaly_types = pd.DataFrame({
                    'Anomaly Type': ['High Sales', 'Low Sales', 'High Price', 'Low Price', 'Unusual Sentiment'],
                    'Count': [
                        anomalies_df['high_sales'].sum(),
                        anomalies_df['low_sales'].sum(),
                        anomalies_df['high_price'].sum(),
                        anomalies_df['low_price'].sum(),
                        anomalies_df['unusual_sentiment'].sum()
                    ]
                })
                
                fig = px.bar(anomaly_types, x='Anomaly Type', y='Count',
                            title='Distribution of Anomaly Types',
                            labels={'Count': 'Number of Events'},
                            color='Count',
                            color_continuous_scale='Oranges')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed anomaly table
                st.markdown("<h3 class='sub-header'>Detailed Anomaly Table</h3>", unsafe_allow_html=True)
                
                display_columns = ['event_date', 'event_category', 'tickets_sold', 'ticket_price', 'sentiment_score',
                                 'high_sales', 'low_sales', 'high_price', 'low_price', 'unusual_sentiment']
                
                anomalies_display = anomalies_df[display_columns].copy()
                anomalies_display['event_date'] = anomalies_display['event_date'].dt.strftime('%Y-%m-%d')
                
                # Format boolean columns for better readability
                for col in ['high_sales', 'low_sales', 'high_price', 'low_price', 'unusual_sentiment']:
                    anomalies_display[col] = anomalies_display[col].apply(lambda x: 'Yes' if x else 'No')
                
                st.dataframe(anomalies_display, use_container_width=True)
                
                # Download button for anomaly data
                csv = anomalies_display.to_csv(index=False)
                st.download_button(
                    label="Download Anomaly Data as CSV",
                    data=csv,
                    file_name="anomaly_data.csv",
                    mime="text/csv",
                )
                
                st.markdown("""
                <div class='info-text'>
                Anomalies are events that deviate significantly from typical patterns. Review these events to identify:
                <ul>
                    <li><b>Opportunities:</b> High sales with high prices or positive sentiment</li>
                    <li><b>Issues:</b> Low sales despite reasonable pricing or negative sentiment</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No anomalies detected in the dataset.")
        else:
            st.warning("No anomaly data available.")

if __name__ == "__main__":
    main()