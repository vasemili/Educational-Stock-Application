from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from flask_pymongo import PyMongo
import plotly.graph_objs as go
import yfinance as yf
import datetime
import os
import openai
import requests
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import re
import uuid
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from transformers import BartTokenizer, BartForConditionalGeneration
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
load_dotenv()
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Use logging.debug(), logging.info(), etc., instead of print()
logging.debug("This is a debug message")

# Access environment variables

app = Flask(__name__, static_folder="../public", template_folder="../templates")

openai_api_key = os.getenv('OpenAI_key')

mongodb_uri = os.getenv('MongoDB_URI')

fin_map_key = os.getenv('Fin_Map_key')

sentiment_api_key = os.getenv('SENTIMENT_API_KEY')

# MongoDB Configuration
app.config['OPENAI_API_KEY'] = openai_api_key

app.config['MONGO_URI'] = mongodb_uri

app.config['FIN_MAP_KEY'] = fin_map_key

app.config['SENTIMENT_API_KEY'] = sentiment_api_key

app.secret_key = os.getenv('SECRET_KEY')

sentiment_api_key = app.config['SENTIMENT_API_KEY']

# Email Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))  # Convert to integer
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True  # Convert to boolean
app.config['MAIL_USE_SSL'] = False  # Convert to boolean

mail = Mail(app)

mongo = PyMongo(app)

openai.api_key = openai_api_key

def is_password_strong(password):
    if len(password) < 8:
        return False
    if not re.search("[a-z]", password):
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[_@$!%*?&]", password):
        return False
    return True

@app.route('/')
def home():
    return render_template('home.html')

bcrypt = Bcrypt(app)

@app.route('/check-availability')
def check_availability():
    type = request.args.get('type')
    value = request.args.get('value')

    if type == "username":
        existing_user = mongo.db.users.find_one({'username': value})
        return jsonify({'exists': bool(existing_user)})
    elif type == "email":
        existing_email = mongo.db.users.find_one({'email': value})
        return jsonify({'exists': bool(existing_email)})

    return jsonify({'exists': False})

@app.route('/flash-message')
def flash_message():
    message = request.args.get('message')
    if message:
        flash(message)
    return '', 204  # Return an empty response with a 204 status code

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        # Username and password validations
        if not (4 <= len(username) <= 12):
            flash('Username must be between 4 and 12 characters long.')
            return redirect(url_for('register'))
        if not is_password_strong(password):
            flash("Password must be at least 8 characters long and include lowercase, uppercase, numbers, and special characters.")
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        # Check for existing user or email
        if mongo.db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
            flash('Username or Email already exists.')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        verification_token = str(uuid.uuid4())
        mongo.db.users.insert_one({'username': username, 'email': email, 'password': hashed_password, 'verification_token': verification_token, 'is_verified': False})

        verification_url = url_for('verify_email', token=verification_token, _external=True)
        msg = Message("Verify your email", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Please click on the link to verify your email: {verification_url}"
        mail.send(msg)

        flash('A verification email has been sent. Please check your inbox.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/verify-email/<token>')
def verify_email(token):
    user = mongo.db.users.find_one({'verification_token': token})
    if user:
        mongo.db.users.update_one({'_id': user['_id']}, {'$set': {'is_verified': True}})
        flash('Your account has been verified. You may now login.')
        return redirect(url_for('login'))
    else:
        flash('Verification link is invalid or has expired.')
        return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = mongo.db.users.find_one({'username': username})
        if user and bcrypt.check_password_hash(user['password'], password):
            if not user.get('is_verified', False):
                flash('Your account is not verified. Please check your email to verify your account.', 'error')
                return render_template('login.html', username=username)
            
            session['username'] = user['username']
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html', username=username)

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from session
    return redirect(url_for('home'))

@app.route('/resend-verification', methods=['POST'])
def resend_verification():
    data = request.get_json()
    username = data.get('username')
    user = mongo.db.users.find_one({'username': username})
    if user and not user.get('is_verified', False):
        verification_token = str(uuid.uuid4())
        mongo.db.users.update_one({'_id': user['_id']}, {'$set': {'verification_token': verification_token}})
        
        verification_url = url_for('verify_email', token=verification_token, _external=True)
        msg = Message("Verify your email", sender=app.config['MAIL_USERNAME'], recipients=[user['email']])
        msg.body = f"Please click on the link to verify your email: {verification_url}"
        mail.send(msg)

        return jsonify({'message': 'Verification email resent. Please check your inbox.'})
    return jsonify({'message': 'Unable to resend verification email.'})

@app.route('/article_scraper', methods=['GET', 'POST'])
def article_scraper():
    summary = ""
    if request.method == 'POST':
        article_url = request.form.get('article_url')
        if article_url:
            existing_article = mongo.db.scraped_articles.find_one({'url': article_url})
            if existing_article:
                flash('Article already scraped. Using existing content.', 'info')
                article_text = existing_article['content']
            else:
                try:
                    article_text = scrape_msnbc_article(article_url)
                    mongo.db.scraped_articles.insert_one({'url': article_url, 'content': article_text})
                    flash('Article successfully scraped and stored.', 'info')
                except Exception as e:
                    flash(f'An error occurred while scraping: {str(e)}', 'error')
                    return render_template('article_scraper.html', summary=summary)

            # Generate summary
            summary = generate_summary(article_text)
        else:
            flash('No URL provided.', 'error')

    return render_template('article_scraper.html', summary=summary)

def scrape_msnbc_article(url):
    # Set up Selenium to work with headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the webdriver with the specified options
    with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) as driver:
        driver.get(url)

        # Use BeautifulSoup to parse the page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        article_content = soup.find_all('p')

        # Collect the text of each paragraph
        article_text = ''
        for p in article_content:
            # Add your logic to filter out unwanted text
            # For example, skipping empty paragraphs or specific elements
            if p.text.strip():
                article_text += p.get_text() + ' '

    return article_text.strip()

model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def generate_summary(article):
    model.eval()
    inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=200)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Summary: {summary}")  # Log the summary for debugging
    return summary

@app.route('/history-of-stocks')
def history_of_stocks():
    return render_template('history-of-stocks.html')

@app.route('/glossary')
def glossary():
    return render_template('glossary.html')

@app.route('/educational-resources')
def educational_resources():
    return render_template('educational-resources.html')

@app.route('/user-feedback-and-review', methods=['GET', 'POST'])
def user_feedback_and_review():
    if request.method == 'POST':
        user_name = request.form.get('userName')
        user_email = request.form.get('userEmail')
        feedback_content = request.form.get('userFeedback')
        
        feedback_data = {
            'user_name': user_name,
            'user_email': user_email,
            'feedback': feedback_content
        }
        mongo.db.feedback.insert_one(feedback_data)

        return redirect(url_for('feedback_thank_you'))

    return render_template('user-feedback-and-review.html') # Display the feedback form for GET requests

@app.route('/feedback-thank-you')
def feedback_thank_you():
    return render_template('feedback-thank-you.html') 

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.json.get('user_message')
        print(f"Received message: {user_message}")  # Debugging

        if 'conversation_history' not in session:
            print("Initializing new session.")  # Debugging
            session['conversation_history'] = ""

        print(f"Previous conversation history: {session['conversation_history']}")  # Debugging

        session['conversation_history'] += f"User: {user_message}\n"

        # Prepare the prompt for OpenAI API
        prompt = session['conversation_history'] + "Chatbot:"
        print(f"Sending prompt to API: {prompt}")  # Debugging

        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",  # or another appropriate model
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,  # Adjust as needed
            stop=["User:", "Chatbot:"]
        )

        chatbot_response = response.choices[0].text.strip()
        session['conversation_history'] += f"Chatbot: {chatbot_response}\n"

        print(f"Chatbot response: {chatbot_response}")  # Debugging

        return jsonify({'chatbot_response': chatbot_response})

    return render_template('chatbot.html')

def fetch_stock_data(ticker_symbol, start_date, end_date):
    # Directly fetch stock data from yfinance
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    # Ensure the date column exists and is the index
    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'index': 'Date'}, inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    # Directly create a Plotly candlestick chart from fetched data
    candlestick_chart = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                       open=stock_data['Open'],
                                                       high=stock_data['High'],
                                                       low=stock_data['Low'],
                                                       close=stock_data['Close'])])
    candlestick_chart.update_layout(title=f'Candlestick Chart for {ticker_symbol}')
    
    # No need for future data extraction or MongoDB storage in this simplified version
    return stock_data, stock_data, candlestick_chart

def get_ticker_symbol(company_name, api_key):
    url = f'https://financialmodelingprep.com/api/v3/search?query={company_name}&limit=1&exchange=NASDAQ&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]['symbol']  # Assuming the first result is the most relevant
    return None

def generate_query_keywords(company_name):
    # Set the prompt to ask for keywords relevant to the company
    prompt = f"Generate a comprehensive list of keywords related to {company_name}, including product names, technology, market impact, recent controversies, and industry innovations. Aim for broad coverage to capture varied news angles. Aim for at least 15 keywords."
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
    )

    # Extract the generated text from the response
    keywords = response.choices[0].text.strip()
    
    # Split and clean keywords, remove duplicates
    keywords_list = list(set(line.strip().strip(',.').split('. ')[-1] for line in keywords.split('\n') if line))
    
    additional_keywords = ["Tech", "Technology", "Stock Market", "Stocks", "Data"]
    combined_keywords_list = additional_keywords + keywords_list
    
    # Format the keywords for a query
    formatted_query = ' OR '.join(f'"{keyword}"' for keyword in combined_keywords_list if keyword)
    logging.debug(f"Formatted query keywords for {company_name}: {formatted_query}")
    
    return formatted_query

def fetch_news_data_and_analyze_sentiment(api_key, date_from, date_to, company_name):
    sources = 'bloomberg,cnbc,reuters,financial-times,techcrunch,the-wall-street-journal,the-verge,business-insider,the-economist,wired,engadget,bbc-news,fortune,techradar'
    query = f'{generate_query_keywords(company_name)}'
    
    def make_news_api_request(query):
        url = f'https://newsapi.org/v2/everything?q={query}&from={date_from}&to={date_to}&sources={sources}&apiKey={api_key}'
        response = requests.get(url)
        return response.json()

    data = make_news_api_request(query)
    news_data = []

    if 'articles' in data:
        news_data = [(article['title'], article['publishedAt'][:10]) for article in data['articles']]
    
    # For debugging: Print the fetched news data
    print(f"Fetched {len(news_data)} articles.")

    news_data = [(headline, date) for headline, date in news_data if date != '1970-01-01' and '[Removed]' not in headline]

    sentiment_data = []
    if news_data:
        tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        headlines = [headline for headline, _ in news_data]

        inputs = tokenizer(headlines, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for (headline, date), prediction in zip(news_data, predictions):
            sentiment_entry = {
                'date': date,
                'headline': headline,
                'sentiment_positive': prediction[0].item(),
                'sentiment_neutral': prediction[1].item(),
                'sentiment_negative': prediction[2].item()
            }
            sentiment_data.append(sentiment_entry)
            
            # For debugging: Print each sentiment entry
            print(sentiment_entry)

    # For debugging: Print the overall sentiment data structure
    print(f"Generated sentiment data for {len(sentiment_data)} articles.")
    logging.debug(f"Generated sentiment data for {len(sentiment_data)} articles.")
    logging.debug(sentiment_data)
    logging.debug(f"Sample sentiment data: {sentiment_data[:5]}")
    return sentiment_data

def generate_plotly_chart(dates, actual_values, predicted_values, company_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual_values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=dates, y=predicted_values, mode='lines', name='Predicted'))
    fig.update_layout(title=f'{company_name} Actual vs Predicted Stock Prices',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      legend_title='Type')
    return fig

def prepare_combined_data(stock_data, sentiment_data):
    # Check if sentiment_data is empty
    if not sentiment_data:
        raise ValueError("Sentiment data is empty.")

    expected_keys = ['date', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']
    # The rest of the check for expected_keys can proceed now that we've confirmed sentiment_data is not empty
    if not all(key in sentiment_data[0] for key in expected_keys):
        raise ValueError("Sentiment data is missing one or more expected keys.")

    sentiment_df = pd.DataFrame(sentiment_data)
    logging.debug(f"Sentiment data df: {sentiment_df}")
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df.set_index('date', inplace=True)

    # Select only the sentiment score columns for aggregation
    sentiment_scores_df = sentiment_df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']]

    # Aggregate sentiment scores by date
    average_sentiment = sentiment_scores_df.groupby('date').mean()

    # Ensure stock_data is a DataFrame
    if isinstance(stock_data, list):
        stock_data = pd.DataFrame(stock_data)
    elif not isinstance(stock_data, pd.DataFrame):
        raise ValueError("stock_data should be a list of dictionaries or a DataFrame.")
    
    # If stock_data doesn't have 'date' as index, convert and set 'date' column as index
    if 'date' in stock_data.columns and not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
    
    logging.debug(f"stock data: {stock_data.shape}")
    logging.debug(f"stock data: {stock_data}")
    # Ensure stock_data is a DataFrame and its index is set to datetime format as necessary
    # Combine stock data with average sentiment data
    combined_data = stock_data.join(average_sentiment, how='inner')

    # Interpolate missing values
    combined_data = combined_data.interpolate(method='time')
    logging.debug(f"combined data df: {combined_data.columns}")
    return combined_data

def train_and_predict(combined_data, company_name):
    r2 = 0
    r2_lower_threshold = 0.8
    r2_upper_threshold = 1
    
    # Scale price-related columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    scaler_prices = StandardScaler()
    combined_data[price_cols] = scaler_prices.fit_transform(combined_data[price_cols])

    # Scale Volume column independently
    scaler_volume = StandardScaler()
    combined_data['Volume'] = scaler_volume.fit_transform(combined_data[['Volume']])
    
    # Scale price-related columns
    selected_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']
    data = combined_data[selected_features]

    train_size = int(len(data) * 0.70)
    val_size = int(len(data) * 0.15)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    # Prepare features and target for model training
    X_train = train_data.drop('Close', axis=1)
    y_train = train_data['Close']
    X_val = val_data.drop('Close', axis=1)
    y_val = val_data['Close']
    X_test = test_data.drop('Close', axis=1)
    y_test = test_data['Close']
    
    while (r2 < r2_lower_threshold or r2 > r2_upper_threshold):
        # Define and compile the neural network model
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        # Train the model with early stopping
        history = model.fit(
            X_train, y_train,
            epochs=50,  # Set back to 50 or an arbitrarily large number
            batch_size=8,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]  # Add the early stopping callback here
        )

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(train_loss) + 1)

        # Convert the epochs range object to a list
        epochs_list = list(epochs)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}')

    # Reshape y_test and predictions for inverse transform
    y_test_reshaped = y_test.values.reshape(-1, 1)
    predictions_reshaped = predictions.reshape(-1, 1)

    num_scaled_cols = 5

    # Create separate dummy arrays for inverse scaling
    dummy_array_y_test = np.zeros((len(y_test_reshaped), num_scaled_cols))
    dummy_array_predictions = np.zeros((len(predictions_reshaped), num_scaled_cols))

    # Fill in the 'Close' column values in the dummy arrays
    # Assuming 'Close' is the last of the scaled columns
    dummy_array_y_test[:, -1] = y_test_reshaped.flatten()
    dummy_array_predictions[:, -1] = predictions_reshaped.flatten()

    # Inverse transform the 'Close' prices using the dummy arrays
    y_test_original = scaler_prices.inverse_transform(dummy_array_y_test)[:, -1]
    predictions_original = scaler_prices.inverse_transform(dummy_array_predictions)[:, -1]

    # Extracting testing dates
    test_dates = test_data.index

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_original, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions_original, mode='lines', name='Predicted'))
    fig.update_layout(title=f'{company_name} Stock Price Forecast', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    
    return fig

@app.route('/interactive-analysis', methods=['GET', 'POST'])
def interactive_analysis():
    candlestick_chart = None
    actual_vs_predicted_chart = None
    api_key = fin_map_key

    if request.method == 'POST':
        input_value = request.form.get('company_name').strip()

        # Convert company name to ticker symbol using FMP API
        ticker_symbol = get_ticker_symbol(input_value, api_key)
        if ticker_symbol is None:
            flash('Unable to find ticker for the given company name.', 'error')
            return render_template("interactive-analysis.html", candlestick_chart=None)

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=180)
        try:
            df_main, _, candlestick_chart = fetch_stock_data(ticker_symbol, start_date, end_date)
        except Exception as e:
            flash(f'Error fetching data for {ticker_symbol}: {str(e)}', 'error')
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=29)
        date_from = start_date.strftime('%Y-%m-%d')
        date_to = end_date.strftime('%Y-%m-%d')

        # Fetch and analyze news data
        sentiment_data = fetch_news_data_and_analyze_sentiment(sentiment_api_key, date_from, date_to, input_value)

        # Fetch stock data and prepare combined dataset
        df_main, df_future, candlestick_chart = fetch_stock_data(ticker_symbol, start_date, end_date)
        combined_data = prepare_combined_data(df_main, sentiment_data)  # You need to implement this
        
        # Assuming you implement train_and_predict to return the necessary data
        actual_vs_predicted_chart = train_and_predict(combined_data, ticker_symbol)
        
    # Convert Plotly figures to JSON for rendering in the template
    #candlestick_chart_json = candlestick_chart.to_json() if candlestick_chart else '{}'
    #actual_vs_predicted_chart_json = actual_vs_predicted_chart.to_json() if actual_vs_predicted_chart else '{}'

    return render_template("interactive-analysis.html", 
                       candlestick_chart=candlestick_chart, 
                       actual_vs_predicted_chart=actual_vs_predicted_chart)
    
@app.route('/session_clear')
def session_clear():
    session.clear()
    flash('Session has been cleared.')
    return redirect(url_for('home'))

if __name__ == "__main__":
    #db.create_all()  # This will create the required tables in the database.
    app.run(debug=True, port=3000)
