
# Educational Stock App

## Description
The Educational Stock App is a comprehensive educational platform for stock market enthusiasts. Built with Flask, JavaScript, HTML, and CSS, it offers a multi-page experience with various features including educational resources, a history of stocks, interactive analysis tools, user authentication, feedback mechanisms, and a unique article summarization feature for financial news.

## Features
- **Educational Resources**: A dedicated page offering valuable learning materials for beginners and advanced traders.
- **Chatbot Integration**: Each page features an interactive chatbot to assist users in navigating the platform and answering queries.
- **Stock History**: Users can explore the historical data of various stocks, visualized effectively.
- **Interactive Analysis**: Tools and graphs for users to analyze stock market trends and patterns, along with an LSTM + Transformer model that uses sentiment analysis and historical company data to do short-term forecasts.
- **User Authentication**: Secure login and registration system with email verification.
- **Feedback System**: Users can provide feedback for continuous improvement of the platform.
- **Article Summarizer**: A feature allowing users to submit URLs of financial articles, which are then summarized for quick insights.

## Getting Started

### Prerequisites
- Python 3.x
- Pip (Python package manager)

### Installation
1. Clone the repository
   git clone https://github.com/vasemili/Educational-Stock-Application

2. Navigate to the project directory
   cd [project directory]

3. Install required Python packages
   Recommended to create and use Python virtual environment prior to installs
   pip install -r requirements.txt

4. Set up environment variables (if applicable)
   Recommened to have .env file that contains (OpenAI_key, MongoDB_URI, Fin_Map_key, SECRET_KEY, MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD, SENTIMENT_API_KEY, MAIL_USE_TLS, and MAIL_USE_SSL)

   The OpenAI_key is used for the chatbot and sentiment analysis. The chatbot uses this key to generate response, and the sentiment analysis uses this key in order to obtain keywords that will later be used for the NewsApi search for news headlines which will be sued for the sentiment analysis using finebert model.
   
   MongoDB_URI is recommended because that is the key you need to store feedback, users login and password info, the web scrapers info obtained from different URLs (this web scraper essentially asks for any URL and obtains the <p> and content of the page and stores it in MongoDB, then we extract this information and use this for abstractive text summarization, this model was fine-tuned specifically for news articles so I recommend feeding in news article sites.)
   
   The Fin_Map_key is the api that basically maps company tickers and company names, without this if the user puts in a company name for their stock analysis the yfinance library won't be able to recognize the company because yfinance requires company tickers. Yfinance is used to dynamically extract specific company's historical static data.
   
   SECRET_KEY is your Flask's secret key, and while it's not necessary it is recommended.
   
   The MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD, MAIL_USE_TLS, MAIL_USE_SSL is used for SMTP emailing people who sign up for the website, if you don't want anyone signing up and logging in the website you don't have to worry about this. But I made it a requisite for people to have their emails verified in order for them to login.
   
   The SENTIMENT_API_KEY is the NewsApis key that is used in order for you to perform the sentiment analysis (the sentiment analysis is important because this is how we use LSTM + Transformer model to forecast (short-term) with higher accuracy)

5. Initialize the database (if applicable)
   python app.py

6. Run the Flask application
   flask run


## Usage
- **Home Page**: Brief description of the home page layout and functionalities.
- **Educational Resources**: Guide on how to access and use educational resources.
- **Stock History Page**: Instructions on how to view and interpret stock history data.
- [Include instructions for other pages/features as needed]

## Contributing
Contributions to the Educational Stock App are welcome. Please follow these steps to contribute:
Please do not try to contribute to this project's repo. You can copy the project and change it
in your own local computer as much as you'd like.

## License
This is just my educational stock application project with incorporations of machine learning
techniques.

## Contact
- Project Link: https://github.com/vasemili/Educational-Stock-Application
- emiliovasquezcarbajalalexander@gmail.com
- (908) 937-8231

