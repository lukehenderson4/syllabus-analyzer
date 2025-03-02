import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime, timedelta
import anthropic
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="DC Weather Oracle",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define cache for data loading to avoid recomputation
@st.cache_data
def load_weather_data(filepath='dc.csv'):
    """Load and preprocess the DC weather data."""
    try:
        # Load CSV file
        df = pd.read_csv(filepath)
        
        # Convert date column to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Extract date components
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        df['day'] = df['DATE'].dt.day
        df['dayofyear'] = df['DATE'].dt.dayofyear
        df['dayofweek'] = df['DATE'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Create season column
        df['season'] = pd.cut(
            df['dayofyear'],
            bins=[0, 79, 171, 263, 366],
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            include_lowest=True
        )
        
        # Fill any missing values in key variables
        for col in ['PRCP', 'SNOW', 'TMAX', 'TMIN']:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Calculate additional features
        df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2  # Average temperature if missing
        df['temp_range'] = df['TMAX'] - df['TMIN']   # Temperature range
        
        # Define a "nice day" as:
        # - Max temp between 65°F and 85°F
        # - Less than 0.1 inches of precipitation
        # - No snow
        df['is_nice_day'] = ((df['TMAX'] >= 65) & 
                             (df['TMAX'] <= 85) & 
                             (df['PRCP'] < 0.1) & 
                             (df['SNOW'] == 0)).astype(int)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def build_weather_models(df):
    """Build and train prediction models for DC weather."""
    models = {}
    
    # Select features for predictive models
    features = ['month', 'day', 'dayofyear', 'is_weekend', 'year']
    
    # Temperature model (TMAX)
    X = df[features]
    y_tmax = df['TMAX']
    
    tmax_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tmax_model.fit(X, y_tmax)
    models['tmax'] = tmax_model
    
    # Temperature model (TMIN)
    y_tmin = df['TMIN']
    tmin_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tmin_model.fit(X, y_tmin)
    models['tmin'] = tmin_model
    
    # Precipitation model
    # Convert to classification: 0 = no rain, 1 = rain
    df['rain_class'] = (df['PRCP'] > 0).astype(int)
    y_rain = df['rain_class']
    
    rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rain_model.fit(X, y_rain)
    models['rain'] = rain_model
    
    # Snow model
    # Convert to classification: 0 = no snow, 1 = snow
    df['snow_class'] = (df['SNOW'] > 0).astype(int)
    y_snow = df['snow_class']
    
    snow_model = RandomForestClassifier(n_estimators=100, random_state=42)
    snow_model.fit(X, y_snow)
    models['snow'] = snow_model
    
    # Nice day model
    y_nice = df['is_nice_day']
    
    nice_model = RandomForestClassifier(n_estimators=100, random_state=42)
    nice_model.fit(X, y_nice)
    models['nice'] = nice_model
    
    # Precipitation amount model (only for days with precipitation)
    rain_df = df[df['PRCP'] > 0]
    if len(rain_df) > 0:
        X_rain = rain_df[features]
        y_rain_amount = rain_df['PRCP']
        
        rain_amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rain_amount_model.fit(X_rain, y_rain_amount)
        models['rain_amount'] = rain_amount_model
    
    # Snow amount model (only for days with snow)
    snow_df = df[df['SNOW'] > 0]
    if len(snow_df) > 0:
        X_snow = snow_df[features]
        y_snow_amount = snow_df['SNOW']
        
        snow_amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
        snow_amount_model.fit(X_snow, y_snow_amount)
        models['snow_amount'] = snow_amount_model
    
    # Time series model for temperature trends
    try:
        # Prophet requires specific column names
        prophet_df = df[['DATE', 'TMAX']].rename(columns={'DATE': 'ds', 'TMAX': 'y'})
        
        # Train model
        prophet_model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        prophet_model.fit(prophet_df)
        models['prophet'] = prophet_model
        
    except Exception as e:
        st.warning(f"Could not build Prophet model: {str(e)}")
    
    return models

def predict_for_date(date, models):
    """Generate weather predictions for a specific date."""
    # Create feature row
    features = pd.DataFrame({
        'month': [date.month],
        'day': [date.day],
        'dayofyear': [date.timetuple().tm_yday],
        'is_weekend': [1 if date.weekday() >= 5 else 0],
        'year': [date.year]
    })
    
    # Make predictions
    predictions = {}
    
    # Temperature predictions
    predictions['tmax'] = models['tmax'].predict(features)[0]
    predictions['tmin'] = models['tmin'].predict(features)[0]
    predictions['tavg'] = (predictions['tmax'] + predictions['tmin']) / 2
    
    # Precipitation predictions
    predictions['rain_prob'] = models['rain'].predict_proba(features)[0][1]
    
    if predictions['rain_prob'] > 0.5 and 'rain_amount' in models:
        predictions['rain_amount'] = models['rain_amount'].predict(features)[0]
    else:
        predictions['rain_amount'] = 0
    
    # Snow predictions
    predictions['snow_prob'] = models['snow'].predict_proba(features)[0][1]
    
    if predictions['snow_prob'] > 0.5 and 'snow_amount' in models:
        predictions['snow_amount'] = models['snow_amount'].predict(features)[0]
    else:
        predictions['snow_amount'] = 0
    
    # Nice day prediction
    predictions['nice_day_prob'] = models['nice'].predict_proba(features)[0][1]
    
    # Include the date in the predictions
    predictions['date'] = date
    
    return predictions

def parse_date_reference(query):
    """Extract date references from natural language query."""
    today = datetime.now().date()
    
    # Handle special cases
    if "today" in query.lower():
        return today
    
    if "tomorrow" in query.lower():
        return today + timedelta(days=1)
    
    if "this weekend" in query.lower():
        # Find the upcoming Saturday
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0:
            days_until_saturday = 7
        return today + timedelta(days=days_until_saturday)
    
    if "next weekend" in query.lower():
        # Find Saturday of next weekend
        days_until_next_saturday = (5 - today.weekday()) % 7 + 7
        return today + timedelta(days=days_until_next_saturday)
    
    # Look for month/day references
    month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)'
    day_pattern = r'(\d{1,2})(st|nd|rd|th)?'
    
    month_match = re.search(month_pattern, query.lower())
    day_match = re.search(day_pattern, query.lower())
    
    if month_match and day_match:
        month_names = ['january', 'february', 'march', 'april', 'may', 'june', 
                      'july', 'august', 'september', 'october', 'november', 'december']
        month_num = month_names.index(month_match.group(1)) + 1
        day_num = int(day_match.group(1))
        
        # Default to current year
        year = today.year
        
        # If the date has already passed this year, assume next year
        if (month_num < today.month) or (month_num == today.month and day_num < today.day):
            year += 1
            
        try:
            return datetime(year, month_num, day_num).date()
        except ValueError:
            # Handle invalid dates (e.g., February 31)
            st.warning(f"Invalid date detected. Using next weekend instead.")
            days_until_saturday = (5 - today.weekday()) % 7
            if days_until_saturday == 0:
                days_until_saturday = 7
            return today + timedelta(days=days_until_saturday)
    
    # Look for day of week references
    days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for i, day in enumerate(days_of_week):
        if day in query.lower():
            days_until = (i - today.weekday()) % 7
            if days_until == 0:  # If today is the mentioned day, assume next week
                days_until = 7
            return today + timedelta(days=days_until)
    
    # Default to next Saturday if no specific date found
    days_until_saturday = (5 - today.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    return today + timedelta(days=days_until_saturday)

def get_weather_response(query, prediction, historical_data, claude_api_key):
    """Generate a natural language response about the weather prediction."""
    # If no API key, generate a simple response
    if not claude_api_key:
        date_str = prediction['date'].strftime("%A, %B %d")
        return generate_simple_response(prediction, date_str)
    
    # Create context with prediction and historical data
    date_str = prediction['date'].strftime("%A, %B %d, %Y")
    
    # Get historical context for similar dates (same month and day)
    month = prediction['date'].month
    day = prediction['date'].day
    
    similar_dates = historical_data[
        (historical_data['month'] == month) & 
        (historical_data['day'] == day)
    ]
    
    historical_context = f"Based on {len(similar_dates)} years of historical data for this date:"
    
    if not similar_dates.empty:
        historical_context += f"""
        - Average high: {similar_dates['TMAX'].mean():.1f}°F
        - Average low: {similar_dates['TMIN'].mean():.1f}°F
        - Days with rain: {(similar_dates['PRCP'] > 0).mean() * 100:.0f}%
        - Days with snow: {(similar_dates['SNOW'] > 0).mean() * 100:.0f}%
        - Nice weather days: {similar_dates['is_nice_day'].mean() * 100:.0f}% of the time
        """
    
    # Prepare message for Claude
    try:
        client = anthropic.Anthropic(api_key=claude_api_key)
        
        prompt = f"""
        You are a friendly weather assistant for Washington DC. I'll give you a weather prediction for a specific date, and I want you to answer the user's question in a conversational way.
        
        Date: {date_str}
        Predicted High Temperature: {prediction['tmax']:.1f}°F
        Predicted Low Temperature: {prediction['tmin']:.1f}°F
        Probability of Rain: {prediction['rain_prob'] * 100:.0f}%
        Predicted Rainfall: {prediction['rain_amount']:.2f} inches
        Probability of Snow: {prediction['snow_prob'] * 100:.0f}%
        Predicted Snowfall: {prediction['snow_amount']:.1f} inches
        Probability of Nice Weather: {prediction['nice_day_prob'] * 100:.0f}%
        
        {historical_context}
        
        User's question: "{query}"
        
        Please give a friendly, helpful response about whether this will be nice weather. Include the prediction but sound natural and conversational. Be specific about the date in your response. If the chance of nice weather is over 70%, be optimistic. If it's under 30%, suggest indoor activities. Between 30-70% be balanced and mention that the weather could go either way.
        
        Your response:
        """
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        st.warning(f"Error connecting to Claude API: {str(e)}")
        return generate_simple_response(prediction, date_str)

def generate_simple_response(prediction, date_str):
    """Generate a simple response without Claude API."""
    response = f"For {date_str}, I predict:"
    
    response += f"\n- High temperature of {prediction['tmax']:.1f}°F and low of {prediction['tmin']:.1f}°F"
    
    if prediction['rain_prob'] > 0.5:
        response += f"\n- Likely rain ({prediction['rain_prob']*100:.0f}% chance) with about {prediction['rain_amount']:.2f} inches of precipitation"
    else:
        response += f"\n- Unlikely to rain ({prediction['rain_prob']*100:.0f}% chance)"
    
    if prediction['snow_prob'] > 0.3:
        response += f"\n- Possible snow ({prediction['snow_prob']*100:.0f}% chance) with about {prediction['snow_amount']:.1f} inches"
    
    if prediction['nice_day_prob'] > 0.7:
        response += f"\n\nOverall, it looks like it will be a very nice day ({prediction['nice_day_prob']*100:.0f}% chance)! Great for outdoor activities."
    elif prediction['nice_day_prob'] > 0.3:
        response += f"\n\nIt might be a nice day ({prediction['nice_day_prob']*100:.0f}% chance), but prepare for weather that could go either way."
    else:
        response += f"\n\nIt's unlikely to be what most consider a 'nice day' ({prediction['nice_day_prob']*100:.0f}% chance). Perhaps plan some indoor activities."
    
    return response

def create_forecast_chart(date, predictions, historical_data):
    """Create a forecast chart for the selected date with historical context."""
    # Get the week containing the target date
    start_date = date - timedelta(days=date.weekday())
    end_date = start_date + timedelta(days=6)
    
    # Create date range for the week
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Generate predictions for each date in the range
    forecast_data = []
    for d in date_range:
        # Convert datetime to date
        d_date = d.date()
        
        # Make prediction for this date
        pred = predict_for_date(d_date, st.session_state.models)
        
        # Add to forecast data
        forecast_data.append({
            'date': d_date,
            'tmax': pred['tmax'],
            'tmin': pred['tmin'],
            'tavg': pred['tavg'],
            'rain_prob': pred['rain_prob'],
            'nice_day_prob': pred['nice_day_prob'],
            'is_target_date': d_date == date
        })
    
    # Create DataFrame
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add high temperature line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['tmax'],
        mode='lines+markers',
        name='High Temp',
        line=dict(color='tomato', width=3),
        marker=dict(
            size=forecast_df['is_target_date'].apply(lambda x: 12 if x else 8),
            color=forecast_df['is_target_date'].apply(lambda x: 'red' if x else 'tomato')
        )
    ))
    
    # Add low temperature line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['tmin'],
        mode='lines+markers',
        name='Low Temp',
        line=dict(color='royalblue', width=3),
        marker=dict(
            size=forecast_df['is_target_date'].apply(lambda x: 12 if x else 8),
            color=forecast_df['is_target_date'].apply(lambda x: 'darkblue' if x else 'royalblue')
        )
    ))
    
    # Add filled area between high and low temps
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['tmax'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['tmin'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.2)',
        showlegend=False
    ))
    
    # Add rain probability as separate y-axis
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['rain_prob'] * 100,  # Convert to percentage
        mode='lines+markers',
        name='Rain Probability (%)',
        line=dict(color='darkgreen', width=2, dash='dash'),
        marker=dict(
            symbol='diamond',
            size=forecast_df['is_target_date'].apply(lambda x: 12 if x else 8)
        ),
        yaxis='y2'
    ))
    
    # Add nice day probability
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['nice_day_prob'] * 100,  # Convert to percentage
        mode='lines+markers',
        name='Nice Day Probability (%)',
        line=dict(color='purple', width=2, dash='dot'),
        marker=dict(
            symbol='circle',
            size=forecast_df['is_target_date'].apply(lambda x: 12 if x else 8)
        ),
        yaxis='y2'
    ))
    
    # Check if we have historical data for this month/day
    month = date.month
    day = date.day
    
    historical_points = historical_data[
        (historical_data['month'] == month) & 
        (historical_data['day'] == day)
    ]
    
    if not historical_points.empty:
        # Get historical high and low ranges
        hist_high_avg = historical_points['TMAX'].mean()
        hist_high_std = historical_points['TMAX'].std()
        hist_low_avg = historical_points['TMIN'].mean()
        hist_low_std = historical_points['TMIN'].std()
        
        # Add historical high temperature range - FIX: removed opacity
        fig.add_trace(go.Scatter(
            x=[date, date],
            y=[hist_high_avg - hist_high_std, hist_high_avg + hist_high_std],
            mode='lines',
            name='Historical High Range',
            line=dict(color='darkred', width=10)
        ))
        
        # Add historical low temperature range - FIX: removed opacity
        fig.add_trace(go.Scatter(
            x=[date, date],
            y=[hist_low_avg - hist_low_std, hist_low_avg + hist_low_std],
            mode='lines',
            name='Historical Low Range',
            line=dict(color='darkblue', width=10)
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Weather Forecast for the Week of {start_date.strftime("%B %d")}',
        xaxis=dict(
            title='Date',
            tickformat='%a, %b %d',  # Format: "Mon, Jan 01"
            tickangle=-45
        ),
        yaxis=dict(
            title='Temperature (°F)',
            range=[
                min(forecast_df['tmin']) - 5,
                max(forecast_df['tmax']) + 5
            ]
        ),
        yaxis2=dict(
            title='Probability (%)',
            range=[0, 100],
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig

def create_historical_monthly_chart(data):
    """Create a chart showing historical monthly averages."""
    # Aggregate data by month
    monthly_data = data.groupby('month').agg({
        'TMAX': 'mean',
        'TMIN': 'mean',
        'PRCP': 'mean',
        'SNOW': 'mean',
        'is_nice_day': 'mean'
    }).reset_index()
    
    # Convert nice day proportion to percentage
    monthly_data['is_nice_day'] = monthly_data['is_nice_day'] * 100
    
    # Create figure with multiple subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"colspan": 2}, None]
        ],
        subplot_titles=("Temperature by Month", "Precipitation by Month", "Nice Days by Month"),
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    
    # Add traces for temperatures
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['TMAX'],
            mode='lines+markers',
            name='Avg High Temp',
            line=dict(color='tomato', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['TMIN'],
            mode='lines+markers',
            name='Avg Low Temp',
            line=dict(color='royalblue', width=3)
        ),
        row=1, col=1
    )
    
    # Add traces for precipitation
    fig.add_trace(
        go.Bar(
            x=monthly_data['month'],
            y=monthly_data['PRCP'],
            name='Avg Precipitation (in)',
            marker_color='darkblue'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_data['month'],
            y=monthly_data['SNOW'],
            name='Avg Snowfall (in)',
            marker_color='lightblue'
        ),
        row=1, col=2, secondary_y=True
    )
    
    # Add trace for nice days
    fig.add_trace(
        go.Bar(
            x=monthly_data['month'],
            y=monthly_data['is_nice_day'],
            name='% Nice Days',
            marker_color='green'
        ),
        row=2, col=1
    )
    
    # Update layout
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.update_layout(
        height=700,
        title='Historical Weather Patterns by Month in DC',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    # Update x-axes
    for i in range(1, 3):
        fig.update_xaxes(
            title_text='Month',
            tickvals=list(range(1, 13)),
            ticktext=months,
            row=1, col=i
        )
    
    fig.update_xaxes(
        title_text='Month',
        tickvals=list(range(1, 13)),
        ticktext=months,
        row=2, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(title_text='Temperature (°F)', row=1, col=1)
    fig.update_yaxes(title_text='Precipitation (in)', row=1, col=2)
    fig.update_yaxes(title_text='Snowfall (in)', secondary_y=True, row=1, col=2)
    fig.update_yaxes(title_text='Percentage of Days', row=2, col=1)
    
    return fig

from plotly.subplots import make_subplots
def create_nice_day_analysis(data):
    """Create analysis of what constitutes a 'nice day' in DC."""
    # Create temperature bands
    temp_bands = pd.cut(data['TMAX'], 
                       bins=[0, 40, 50, 60, 70, 80, 90, 100, 110], 
                       labels=['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '>100'])
    
    # Calculate percentage of nice days in each temperature band
    nice_by_temp = data.groupby(temp_bands)['is_nice_day'].mean() * 100
    
    # Create precipitation bands
    precip_bands = pd.cut(data['PRCP'], 
                         bins=[0, 0.01, 0.1, 0.25, 0.5, 1, 5], 
                         labels=['0', '0-0.01', '0.01-0.1', '0.1-0.25', '0.25-0.5', '>0.5'])
    
    # Calculate percentage of nice days in each precipitation band
    nice_by_precip = data.groupby(precip_bands)['is_nice_day'].mean() * 100
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("'Nice Day' Percentage by Temperature", "'Nice Day' Percentage by Precipitation"),
        vertical_spacing=0.15
    )
    
    # Add bar charts
    fig.add_trace(
        go.Bar(
            x=nice_by_temp.index,
            y=nice_by_temp.values,
            name='By Temperature',
            marker_color='orange'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=nice_by_precip.index,
            y=nice_by_precip.values,
            name='By Precipitation',
            marker_color='blue'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title='Analysis of "Nice Day" Conditions in DC',
        showlegend=False
    )
    
    fig.update_yaxes(title_text='% of Nice Days', range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text='% of Nice Days', range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text='Maximum Temperature (°F)', row=1, col=1)
    fig.update_xaxes(title_text='Precipitation (inches)', row=2, col=1)
    
    return fig

def main():
    # Page title
    st.title("☀️ DC Weather Oracle")
    st.markdown("### Historical Analysis & Future Predictions for Washington DC")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    
    # Sidebar for API key and data loading
    with st.sidebar:
        st.title("Settings")
        claude_api_key = st.text_input("Claude API Key (optional)", 
                                     type="password", 
                                     help="For natural language responses")
        
        if st.button("Load Weather Data & Build Models"):
            with st.spinner("Loading weather data..."):
                weather_data = load_weather_data('dc.csv')
                
                if weather_data is not None:
                    st.session_state.weather_data = weather_data
                    st.session_state.data_loaded = True
                    
                    # Build models
                    with st.spinner("Building prediction models (this may take a minute)..."):
                        st.session_state.models = build_weather_models(weather_data)
                    
                    st.success("✅ Data loaded and models built successfully!")
                else:
                    st.error("❌ Failed to load weather data.")
    
    # Main content
    if not st.session_state.data_loaded:
        # Show welcome screen
        st.info("👋 Welcome to the DC Weather Oracle! Please load the weather data using the sidebar button to get started.")
        
        st.markdown("""
        ### What can this app do?
        
        - **Predict weather** for any date in DC
        - **Answer natural language questions** about upcoming weather
        - **Analyze historical patterns** in temperature, precipitation, and more
        - **Define what makes a "nice day"** in DC based on historical data
        
        To get started, press the "Load Weather Data & Build Models" button in the sidebar. This will load the historical data and build the predictive models.
        """)
        
        st.image("https://images.unsplash.com/photo-1617581629397-0066ed371547", caption="Washington DC in the Spring")
        
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Weather Prediction", "Historical Analysis", "About"])
    
    # Weather Prediction Tab
    with tab1:
        st.header("🔮 Predict DC Weather")
        
        # Natural language query
        st.subheader("Ask about the weather")
        query = st.text_input("Ask about upcoming weather in DC:", 
                           placeholder="Will it be nice in DC this weekend?",
                           help="Try phrases like 'this weekend', 'next Saturday', or specific dates")
        
        if query:
            with st.spinner("🤔 Analyzing your question..."):
                # Parse date from query
                target_date = parse_date_reference(query)
                
                # Generate prediction
                prediction = predict_for_date(target_date, st.session_state.models)
                
                # Get response
                response = get_weather_response(query, prediction, st.session_state.weather_data, claude_api_key)
                
                # Display results
                st.markdown(f"### Weather forecast for {target_date.strftime('%A, %B %d, %Y')}")
                
                # Use columns for metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Temperature", f"{prediction['tmax']:.1f}°F")
                with col2:
                    st.metric("Low Temperature", f"{prediction['tmin']:.1f}°F")
                with col3:
                    st.metric("Nice Day Probability", f"{prediction['nice_day_prob']*100:.0f}%")
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Rain Probability", f"{prediction['rain_prob']*100:.0f}%")
                    if prediction['rain_prob'] > 0.5:
                        st.metric("Predicted Rainfall", f"{prediction['rain_amount']:.2f} in")
                with col5:
                    st.metric("Snow Probability", f"{prediction['snow_prob']*100:.0f}%")
                    if prediction['snow_prob'] > 0.3:
                        st.metric("Predicted Snowfall", f"{prediction['snow_amount']:.1f} in")
                
                # Show Claude's response
                st.markdown("### Weather Oracle Says:")
                st.info(response)
                
                # Show weekly forecast chart
                st.subheader("Weekly Forecast")
                forecast_chart = create_forecast_chart(target_date, prediction, st.session_state.weather_data)
                st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Or select a specific date
        st.markdown("---")
        st.subheader("Or select a specific date")
        
        # Date selection
        max_date = datetime.now().date() + timedelta(days=365)  # Predict up to 1 year in advance
        selected_date = st.date_input("Select date:", 
                                   min_value=datetime.now().date(),
                                   max_value=max_date,
                                   help="Select a date to get a weather prediction")
        
        if st.button("Get Prediction"):
            with st.spinner("Generating prediction..."):
                # Make prediction
                prediction = predict_for_date(selected_date, st.session_state.models)
                
                # Display results
                st.markdown(f"### Weather forecast for {selected_date.strftime('%A, %B %d, %Y')}")
                
                # Use columns for metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Temperature", f"{prediction['tmax']:.1f}°F")
                with col2:
                    st.metric("Low Temperature", f"{prediction['tmin']:.1f}°F")
                with col3:
                    st.metric("Nice Day Probability", f"{prediction['nice_day_prob']*100:.0f}%")
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Rain Probability", f"{prediction['rain_prob']*100:.0f}%")
                    if prediction['rain_prob'] > 0.5:
                        st.metric("Predicted Rainfall", f"{prediction['rain_amount']:.2f} in")
                with col5:
                    st.metric("Snow Probability", f"{prediction['snow_prob']*100:.0f}%")
                    if prediction['snow_prob'] > 0.3:
                        st.metric("Predicted Snowfall", f"{prediction['snow_amount']:.1f} in")
                
                # Generate simple response (without Claude)
                response = generate_simple_response(prediction, selected_date.strftime("%A, %B %d"))
                
                # Show response
                st.markdown("### Weather Oracle Says:")
                st.info(response)
                
                # Show weekly forecast chart
                st.subheader("Weekly Forecast")
                forecast_chart = create_forecast_chart(selected_date, prediction, st.session_state.weather_data)
                st.plotly_chart(forecast_chart, use_container_width=True)
    
    # Historical Analysis Tab
    with tab2:
        st.header("📊 Historical Weather Patterns")
        
        # Monthly weather patterns
        st.subheader("Monthly Weather Patterns")
        monthly_chart = create_historical_monthly_chart(st.session_state.weather_data)
        st.plotly_chart(monthly_chart, use_container_width=True)
        
        # What makes a nice day?
        st.subheader("What Makes a 'Nice Day' in DC?")
        st.markdown("""
        The app defines a 'nice day' as having:
        - High temperature between 65°F and 85°F
        - Less than 0.1 inches of precipitation
        - No snow
        
        The charts below show how different weather conditions affect the likelihood of a 'nice day':
        """)
        
        nice_day_chart = create_nice_day_analysis(st.session_state.weather_data)
        st.plotly_chart(nice_day_chart, use_container_width=True)
        
        # Season analysis
        st.subheader("Weather by Season")
        
        # Calculate seasonal statistics
        season_data = st.session_state.weather_data.groupby('season').agg({
            'TMAX': 'mean',
            'TMIN': 'mean',
            'PRCP': 'mean',
            'SNOW': 'mean',
            'is_nice_day': 'mean'
        }).reset_index()
        
        # Sort by natural season order
        season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
        season_data['order'] = season_data['season'].map(season_order)
        season_data = season_data.sort_values('order').drop('order', axis=1)
        
        # Convert nice day proportion to percentage
        season_data['is_nice_day'] = season_data['is_nice_day'] * 100
        
        # Display as table
        st.dataframe(season_data.round(2).set_index('season'), use_container_width=True)
        
        # Best times to visit
        nice_days_by_month = st.session_state.weather_data.groupby('month')['is_nice_day'].mean()
        best_month = nice_days_by_month.idxmax()
        worst_month = nice_days_by_month.idxmin()
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        st.markdown(f"""
        ### Best Times to Visit DC
        
        Based on historical data, the best month to visit DC is **{month_names[best_month-1]}** with 
        **{nice_days_by_month.max()*100:.1f}%** nice days.
        
        The worst month is **{month_names[worst_month-1]}** with only 
        **{nice_days_by_month.min()*100:.1f}%** nice days.
        """)
        
        # Show nice day percentage by month
        fig = px.bar(
            x=list(range(1, 13)),
            y=nice_days_by_month.values * 100,
            labels={'x': 'Month', 'y': 'Percentage of Nice Days'},
            title='Percentage of Nice Days by Month'
        )
        
        fig.update_layout(
            xaxis=dict(
                tickvals=list(range(1, 13)),
                ticktext=month_names
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # About tab
    with tab3:
        st.header("ℹ️ About DC Weather Oracle")
        
        st.markdown("""
        ### How it Works
        
        This app uses historical weather data from NOAA for Washington DC to predict future weather patterns. 
        The main components are:
        
        1. **Data Processing**: Historical weather data is loaded and preprocessed, including feature engineering
           to identify "nice days" based on temperature and precipitation criteria.
        
        2. **Machine Learning Models**: Several models are built to predict different aspects of the weather:
           - Random Forest Regression models for temperature predictions
           - Random Forest Classification models for rain and snow probability
           - Random Forest Regression models for precipitation and snowfall amounts
        
        3. **Natural Language Interface**: Claude AI is used to parse date references from natural language
           queries and generate human-like responses explaining the weather predictions.
        
        4. **Visualization**: Interactive charts and dashboards display both historical patterns and
           forecasts to help understand DC's weather patterns.
        
        ### Data Sources
        
        The weather data comes from NOAA's National Centers for Environmental Information (NCEI) and
        contains daily records of various weather parameters for Washington DC, including:
        
        - Maximum and minimum temperatures
        - Precipitation
        - Snowfall
        - Other weather indicators
        
        ### Technical Details
        
        The app uses:
        - Python with Streamlit for the web interface
        - Scikit-learn for machine learning models
        - Plotly for interactive visualizations
        - Anthropic's Claude API for natural language processing
        
        ### Limitations
        
        This weather prediction app has several limitations:
        
        - It relies on historical patterns, which may change due to climate change
        - It doesn't incorporate current weather forecasts or real-time data
        - The models are relatively simple compared to modern meteorological forecasting
        - Accuracy decreases the further into the future you try to predict
        
        Remember that this is a demonstration app, not a replacement for professional weather forecasts!
        """)

if __name__ == "__main__":
    main()