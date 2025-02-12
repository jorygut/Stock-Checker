import requests
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
import json
import time
from selenium.webdriver.chrome.options import Options
import os
from fuzzywuzzy import fuzz
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import yfinance as yf

# Extract article info from news API
def fetch_articles(name):
    API_KEY = "7c41beb12c4e426488da09b8200c61d5"
    BASE_URL = "https://newsapi.org/v2/everything"

    params = {
        "q": name,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 100,
        "apiKey": API_KEY
    }
    all_articles = []
    page = 1

    while True:
        print(f"Fetching page {page}...")
        params["page"] = page
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                if not articles:
                    break
                
                for idx, article in enumerate(articles, start=1 + (page - 1) * 100):
                    return_dict = {
                        "title": article.get('title'),
                        "source": article.get('source', {}).get('name'),
                        "publishedAt": article.get('publishedAt'),
                        "url": article.get('url')
                    }
                    all_articles.append(return_dict)
                    print(f"{idx}. {article.get('title')}")
                    print(f"   Source: {article.get('source', {}).get('name')}")
                    print(f"   Published At: {article.get('publishedAt')}")
                    print(f"   URL: {article.get('url')}\n")
                
                page += 1
            else:
                print("No articles found or an error occurred in the API response.")
                break
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break

    print(f"\nTotal articles retrieved: {len(all_articles)}")
    return all_articles
# Scrape article text
def scrape_articles(articles):
    driver = webdriver.Chrome()
    for a in articles:
        try:
            driver.get(a['url'])

            page_text = driver.find_element(By.TAG_NAME, 'body').text
            a['text'] = page_text
        except:
            a['text'] = ""
    return articles
# Chunk text 300 characters
def chunk_text(text):
    chunks= []
    i = 0
    end_text = False
    while not end_text:
        cur_chunk = text[i:(i + 300)]
        chunks.append(cur_chunk)
        if len(cur_chunk) < 300:
            break
        i += 300
    return chunks

# Extract emotional weights
def analyze_text_emotions(articles):
    nltk.download('vader_lexicon')

    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    for a in articles:
        all_scores = []
        chunked_text = chunk_text(a['text'])
        for c in chunked_text:
            score = classifier(c)
            all_scores.append(score)
        a['scores'] = all_scores

    for art in articles:
        total_dict = {}
        for i in art['scores']:
            for l in i[0]:
                if l['label'] in total_dict:
                    total_dict[l['label']] += l['score']
                    total_dict[f'{l["label"]}_count'] += 1
                else:
                    total_dict[l['label']] = l['score']
                    total_dict[f'{l["label"]}_count'] = 1
        avg_dict = {}
        for t in total_dict:
            if "count" not in t:
                avg_dict[t] = total_dict[t] / total_dict[f'{t}_count']
        art.pop('scores', None)
        art.update(avg_dict)

    df = pd.DataFrame(articles)
    return df

# Get average emotional weights
def analyze_emotional_weights(df):
    emotions = []

    for i in df.columns:
        if i != "title" and i != "source" and i != "publishedAt" and i != "url" and i != "text":
            emotions.append(i)
    emotion_averages = df[emotions].mean()
    print(emotion_averages)

    # Graph emotions
    plt.figure(figsize=(10, 6))
    emotion_averages.plot(kind='bar')

    plt.title('Average Emotion Scores', fontsize=16)
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show graph
    plt.tight_layout()
    plt.show()

# Gather financial statements
def gather_financials(ticker ):
    ticker_symbol = ticker

    ticker = yf.Ticker(ticker_symbol)

    income_statement = ticker.financials
    balance_sheet = ticker.balance_sheet
    cash_flow = ticker.cashflow
    historical_prices = ticker.history(period="5y")

    return income_statement, balance_sheet, cash_flow, historical_prices

# Extract key ratios from income statement
def analyze_income_statement(data):
    df = pd.DataFrame(data)
    ratios = {}

    for year in df.columns:
        gross_profit = df.loc['Gross Profit', year]
        total_revenue = df.loc['Total Revenue', year]
        gross_margin = gross_profit / total_revenue if total_revenue != 0 else None

        net_income = df.loc['Net Income', year]
        net_profit_margin = net_income / total_revenue if total_revenue != 0 else None

        ebitda = df.loc['EBITDA', year]
        ebitda_margin = ebitda / total_revenue if total_revenue != 0 else None

        operating_income = df.loc['Operating Income', year]
        operating_margin = operating_income / total_revenue if total_revenue != 0 else None

        net_income = df.loc['Net Income', year]


        eps = df.loc['Basic EPS', year]

        ebit = df.loc['EBIT', year]
        interest_expense = df.loc['Interest Expense', year]
        interest_coverage = ebit / interest_expense if interest_expense and interest_expense != 0 else None


        ratios[year] = {
            "Gross Margin": gross_margin,
            "Net Profit Margin": net_profit_margin,
            "EBITDA Margin": ebitda_margin,
            "Operating Margin": operating_margin,
            "Earnings Per Share (EPS)": eps,
            "Interest Coverage Ratio": interest_coverage,
        }

    ratios_df = pd.DataFrame(ratios)
    return ratios_df

# Extract key ratios from balance sheet
def analyze_balance_sheet(balance_sheet):
    for i in balance_sheet.columns:
        dtoe_ratio = balance_sheet[i]['Total Liabilities Net Minority Interest'] / balance_sheet[i]['Total Equity Gross Minority Interest']
        debt_ratio = balance_sheet[i]['Total Liabilities Net Minority Interest'] / balance_sheet[i]['Total Assets']
        working_cap = balance_sheet[i]['Current Assets'] - balance_sheet[i]['Current Liabilities']
        print(f"Debt-To-Equity Ratio for {i}: {dtoe_ratio}")
        print(f"Debt Ratio for {i}: {debt_ratio}")
        print(f"Working Capital for {i}: {working_cap}")

# Extract key ratios from cash flow statement
def analyze_cash_flow(cash_flow):
    for i in cash_flow.columns:
        ofc_margin = cash_flow[i]['Operating Cash Flow'] / cash_flow[i]['Changes In Cash']
        fcf = cash_flow[i]['Operating Cash Flow'] / cash_flow[i]['Capital Expenditure']

        print(f"OFC Margin for {i}: {ofc_margin}")
        print(f"Free Cash Flow for {i}: {fcf}")

# Calculate Relative Strength Index
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Graph prices and rolling averages
def analyze_prices(data):

    data['Daily Range'] = data['High'] - data['Low']

    data['20-Day MA'] = data['Close'].rolling(window=20).mean()
    data['50-Day MA'] = data['Close'].rolling(window=50).mean()

    data['RSI'] = calculate_rsi(data)

    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['20-Day MA'], label='20-Day MA', linestyle='--')
    plt.plot(data['50-Day MA'], label='50-Day MA', linestyle='--')
    plt.title("Stock Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.bar(data.index, data['Volume'], label='Volume', color='gray')
    plt.title("Volume Analysis")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid()
    plt.show()

    return data

def main():
    name = ""
    ticker = ""
    articles = fetch_articles(name)
    articles_with_text = scrape_articles(articles)
    df_analyzed = analyze_text_emotions(articles_with_text)
    analyze_emotional_weights(df_analyzed)

    income_statement, balance_sheet, cash_flow, historical_prices = gather_financials(ticker)

    income_ratios_df = analyze_income_statement(income_statement)
    print(income_ratios_df)

    analyze_balance_sheet(balance_sheet)

    analyze_cash_flow(cash_flow)

    analyzed_prices = analyze_prices(historical_prices)
    print(analyzed_prices)


if __name__ == "__main__":
    main()
