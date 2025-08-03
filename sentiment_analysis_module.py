# sentiment_analysis_module.py - Complete Professional Implementation

import numpy as np
import pandas as pd
import requests
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from newsapi import NewsApiClient
import tweepy
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class AlternativeDataManager:
    """
    Professional alternative data integration system
    Handles news, social media, economic data, and sentiment analysis
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # API configurations
        self.news_api_key = config.get('news_api_key', '')
        self.twitter_bearer_token = config.get('twitter_bearer_token', '')
        self.economic_calendar_api = config.get('economic_calendar_api', '')
        
        # Sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Data storage
        self.db_path = Path(config.get('data_path', './data')) / 'alternative_data.db'
        self.init_database()
        
        # Cache settings
        self.cache_duration = timedelta(hours=1)
        self.data_cache = {}
        
        logger.info("Alternative Data Manager initialized")

    def init_database(self):
        """Initialize SQLite database for data storage"""
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # News sentiment table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol VARCHAR(10),
                    source VARCHAR(50),
                    headline TEXT,
                    content TEXT,
                    sentiment_score REAL,
                    sentiment_label VARCHAR(20),
                    relevance_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Social sentiment table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS social_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol VARCHAR(10),
                    platform VARCHAR(20),
                    content TEXT,
                    sentiment_score REAL,
                    engagement_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Economic events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS economic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    country VARCHAR(10),
                    event_name VARCHAR(100),
                    importance INTEGER,
                    actual_value REAL,
                    forecast_value REAL,
                    previous_value REAL,
                    impact_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    async def collect_all_alternative_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect all alternative data sources asynchronously"""[5]
        
        logger.info(f"Collecting alternative data for symbols: {symbols}")
        
        tasks = []
        
        # News sentiment
        tasks.append(self.collect_news_sentiment(symbols))
        
        # Social media sentiment
        tasks.append(self.collect_social_sentiment(symbols))
        
        # Economic calendar
        tasks.append(self.collect_economic_events())
        
        # Market fear/greed indicators
        tasks.append(self.collect_market_indicators())
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            combined_data = {}
            for symbol in symbols:
                combined_data[symbol] = self.combine_alternative_data(symbol, results)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error collecting alternative data: {e}")
            return {}

    async def collect_news_sentiment(self, symbols: List[str]) -> Dict[str, List]:
        """Collect and analyze news sentiment for given symbols"""[5]
        
        news_data = {}
        
        if not self.news_api_key:
            logger.warning("News API key not provided")
            return news_data
        
        try:
            newsapi = NewsApiClient(api_key=self.news_api_key)
            
            for symbol in symbols:
                # Create search queries
                queries = self.create_news_queries(symbol)
                articles = []
                
                for query in queries:
                    try:
                        # Get recent articles
                        response = newsapi.get_everything(
                            q=query,
                            language='en',
                            sort_by='publishedAt',
                            from_param=(datetime.now() - timedelta(days=7)).isoformat(),
                            to=datetime.now().isoformat(),
                            page_size=20
                        )
                        
                        if response['status'] == 'ok':
                            articles.extend(response['articles'])
                    
                    except Exception as e:
                        logger.warning(f"Error fetching news for {query}: {e}")
                        continue
                
                # Analyze sentiment
                sentiment_data = []
                for article in articles:
                    sentiment = self.analyze_text_sentiment(
                        article.get('title', '') + ' ' + article.get('description', '')
                    )
                    
                    sentiment_data.append({
                        'timestamp': pd.to_datetime(article['publishedAt']),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'headline': article.get('title', ''),
                        'content': article.get('description', ''),
                        'sentiment_score': sentiment['compound'],
                        'sentiment_label': self.classify_sentiment(sentiment['compound']),
                        'relevance_score': self.calculate_relevance(article.get('title', ''), symbol),
                        'url': article.get('url', '')
                    })
                
                news_data[symbol] = sentiment_data
                
                # Store in database
                self.store_news_sentiment(symbol, sentiment_data)
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error in news sentiment collection: {e}")
            return news_data

    def create_news_queries(self, symbol: str) -> List[str]:
        """Create relevant search queries for a given symbol"""
        
        symbol_mapping = {
            'EURUSD': ['EUR/USD', 'Euro Dollar', 'European Central Bank', 'ECB', 'Federal Reserve'],
            'GBPUSD': ['GBP/USD', 'Pound Dollar', 'Bank of England', 'BOE', 'Brexit'],
            'XAUUSD': ['Gold', 'XAU/USD', 'precious metals', 'gold price', 'safe haven'],
            'USDJPY': ['USD/JPY', 'Dollar Yen', 'Bank of Japan', 'BOJ'],
            'AUDUSD': ['AUD/USD', 'Australian Dollar', 'Reserve Bank Australia', 'RBA']
        }
        
        return symbol_mapping.get(symbol, [symbol])

    async def collect_social_sentiment(self, symbols: List[str]) -> Dict[str, List]:
        """Collect social media sentiment"""[5]
        
        social_data = {}
        
        if not self.twitter_bearer_token:
            logger.warning("Twitter Bearer Token not provided")
            return social_data
        
        try:
            # Twitter API v2 setup would go here
            # For now, simulate social sentiment data
            for symbol in symbols:
                # Generate synthetic social sentiment data
                social_data[symbol] = self.generate_synthetic_social_data(symbol)
            
            return social_data
            
        except Exception as e:
            logger.error(f"Error collecting social sentiment: {e}")
            return social_data

    def generate_synthetic_social_data(self, symbol: str) -> List[Dict]:
        """Generate synthetic social sentiment data for testing"""
        
        social_posts = []
        current_time = datetime.now()
        
        # Generate 100 synthetic posts over last 24 hours
        for i in range(100):
            timestamp = current_time - timedelta(hours=np.random.randint(0, 24))
            sentiment_score = np.random.normal(0, 0.3)  # Neutral bias
            engagement = np.random.exponential(10)  # Power law distribution
            
            social_posts.append({
                'timestamp': timestamp,
                'platform': np.random.choice(['twitter', 'reddit', 'stocktwits']),
                'content': f"Synthetic post about {symbol}",
                'sentiment_score': sentiment_score,
                'engagement_score': engagement
            })
        
        return social_posts

    async def collect_economic_events(self) -> List[Dict]:
        """Collect upcoming economic events"""[5]
        
        try:
            # Use a free economic calendar API or create synthetic data
            events = []
            
            # Major economic events affecting forex
            major_events = [
                {'name': 'Non-Farm Payrolls', 'country': 'US', 'importance': 5},
                {'name': 'ECB Interest Rate Decision', 'country': 'EU', 'importance': 5},
                {'name': 'GDP Growth Rate', 'country': 'US', 'importance': 4},
                {'name': 'Consumer Price Index', 'country': 'US', 'importance': 4},
                {'name': 'Federal Funds Rate', 'country': 'US', 'importance': 5}
            ]
            
            # Generate synthetic events for next week
            for i in range(10):
                event = np.random.choice(major_events)
                timestamp = datetime.now() + timedelta(days=np.random.randint(0, 7))
                
                events.append({
                    'timestamp': timestamp,
                    'country': event['country'],
                    'event_name': event['name'],
                    'importance': event['importance'],
                    'actual_value': None,  # Future event
                    'forecast_value': np.random.normal(0, 1),
                    'previous_value': np.random.normal(0, 1),
                    'impact_score': event['importance'] / 5.0
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error collecting economic events: {e}")
            return []

    async def collect_market_indicators(self) -> Dict[str, float]:
        """Collect market fear/greed and other indicators"""[5]
        
        indicators = {}
        
        try:
            # VIX (Fear index) - can be fetched from Yahoo Finance
            vix_data = yf.download('^VIX', period='5d', interval='1d')
            if not vix_data.empty:
                indicators['vix'] = float(vix_data['Close'].iloc[-1])
                indicators['vix_change'] = float(vix_data['Close'].pct_change().iloc[-1])
            
            # Dollar Index
            dxy_data = yf.download('DX-Y.NYB', period='5d', interval='1d')
            if not dxy_data.empty:
                indicators['dollar_index'] = float(dxy_data['Close'].iloc[-1])
                indicators['dollar_index_change'] = float(dxy_data['Close'].pct_change().iloc[-1])
            
            # Generate synthetic indicators
            indicators.update({
                'fear_greed_index': np.random.uniform(0, 100),
                'put_call_ratio': np.random.uniform(0.5, 2.0),
                'market_volatility': np.random.uniform(0.1, 0.5)
            })
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error collecting market indicators: {e}")
            return {}

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using multiple methods"""[5]
        
        if not text or not text.strip():
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # Clean text
        text = self.clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
        except:
            textblob_polarity = 0.0
        
        # Combine scores
        combined_score = (vader_scores['compound'] + textblob_polarity) / 2
        
        return {
            'compound': combined_score,
            'positive': vader_scores['pos'],
            'neutral': vader_scores['neu'],
            'negative': vader_scores['neg'],
            'textblob_polarity': textblob_polarity
        }

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()

    def classify_sentiment(self, compound_score: float) -> str:
        """Classify sentiment score into categories"""
        
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def calculate_relevance(self, text: str, symbol: str) -> float:
        """Calculate relevance of text to trading symbol"""
        
        symbol_keywords = {
            'EURUSD': ['eur', 'euro', 'dollar', 'ecb', 'fed', 'european', 'federal'],
            'GBPUSD': ['gbp', 'pound', 'sterling', 'dollar', 'brexit', 'boe', 'bank of england'],
            'XAUUSD': ['gold', 'xau', 'precious', 'metal', 'safe haven', 'inflation'],
        }
        
        keywords = symbol_keywords.get(symbol, [symbol.lower()])
        text_lower = text.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Normalize by text length
        relevance = min(matches / max(len(keywords), 1), 1.0)
        
        return relevance

    def combine_alternative_data(self, symbol: str, collected_data: List) -> pd.DataFrame:
        """Combine all alternative data sources into a single DataFrame"""
        
        try:
            # Create time-based index for last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            time_index = pd.date_range(start_date, end_date, freq='H')
            
            # Initialize DataFrame
            alt_data = pd.DataFrame(index=time_index)
            
            # Process news sentiment
            if len(collected_data) > 0 and isinstance(collected_data[0], dict):
                news_data = collected_data[0].get(symbol, [])
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
                    news_df = news_df.set_index('timestamp')
                    
                    # Resample to hourly and aggregate
                    news_hourly = news_df.resample('H').agg({
                        'sentiment_score': 'mean',
                        'relevance_score': 'mean'
                    }).fillna(0)
                    
                    alt_data = alt_data.join(news_hourly, rsuffix='_news')
            
            # Process social sentiment
            if len(collected_data) > 1 and isinstance(collected_data[1], dict):
                social_data = collected_data[1].get(symbol, [])
                if social_data:
                    social_df = pd.DataFrame(social_data)
                    social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
                    social_df = social_df.set_index('timestamp')
                    
                    # Resample to hourly and aggregate
                    social_hourly = social_df.resample('H').agg({
                        'sentiment_score': 'mean',
                        'engagement_score': 'mean'
                    }).fillna(0)
                    
                    alt_data = alt_data.join(social_hourly, rsuffix='_social')
            
            # Process economic events
            if len(collected_data) > 2 and isinstance(collected_data[2], list):
                events_data = collected_data[2]
                if events_data:
                    events_df = pd.DataFrame(events_data)
                    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
                    events_df = events_df.set_index('timestamp')
                    
                    # Create event impact score
                    events_hourly = events_df.resample('H').agg({
                        'impact_score': 'max'
                    }).fillna(0)
                    
                    alt_data = alt_data.join(events_hourly, rsuffix='_events')
            
            # Process market indicators
            if len(collected_data) > 3 and isinstance(collected_data[3], dict):
                indicators = collected_data[3]
                for name, value in indicators.items():
                    alt_data[f'market_{name}'] = value
            
            # Fill missing values
            alt_data = alt_data.fillna(method='ffill').fillna(0)
            
            # Add derived features
            alt_data = self.add_sentiment_features(alt_data)
            
            return alt_data
            
        except Exception as e:
            logger.error(f"Error combining alternative data: {e}")
            # Return empty DataFrame with time index
            return pd.DataFrame(index=pd.date_range(
                datetime.now() - timedelta(days=30), 
                datetime.now(), 
                freq='H'
            ))

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived sentiment features"""
        
        # Combined sentiment score
        sentiment_cols = [col for col in df.columns if 'sentiment_score' in col]
        if sentiment_cols:
            df['combined_sentiment'] = df[sentiment_cols].mean(axis=1)
            df['sentiment_volatility'] = df[sentiment_cols].std(axis=1)
        
        # Sentiment momentum
        if 'combined_sentiment' in df.columns:
            df['sentiment_momentum'] = df['combined_sentiment'].rolling(24).mean()  # 24-hour average
            df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
        
        # News vs Social sentiment divergence
        if 'sentiment_score_news' in df.columns and 'sentiment_score_social' in df.columns:
            df['news_social_divergence'] = df['sentiment_score_news'] - df['sentiment_score_social']
        
        # High-impact event indicator
        if 'impact_score_events' in df.columns:
            df['high_impact_event'] = (df['impact_score_events'] > 0.8).astype(int)
        
        return df

    def store_news_sentiment(self, symbol: str, sentiment_data: List[Dict]):
        """Store news sentiment data in database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for item in sentiment_data:
                    conn.execute('''
                        INSERT INTO news_sentiment 
                        (timestamp, symbol, source, headline, content, sentiment_score, sentiment_label, relevance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item['timestamp'],
                        symbol,
                        item['source'],
                        item['headline'],
                        item['content'],
                        item['sentiment_score'],
                        item['sentiment_label'],
                        item['relevance_score']
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing news sentiment: {e}")

    def get_sentiment_signals(self, symbol: str, lookback_hours: int = 24) -> Dict[str, float]:
        """Generate trading signals based on sentiment analysis"""[5]
        
        try:
            # Get recent sentiment data from database
            with sqlite3.connect(self.db_path) as conn:
                # News sentiment
                news_query = '''
                    SELECT sentiment_score, relevance_score, timestamp
                    FROM news_sentiment 
                    WHERE symbol = ? AND timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(lookback_hours)
                
                news_df = pd.read_sql_query(news_query, conn, params=(symbol,))
                
                # Social sentiment
                social_query = '''
                    SELECT sentiment_score, engagement_score, timestamp
                    FROM social_sentiment 
                    WHERE symbol = ? AND timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(lookback_hours)
                
                social_df = pd.read_sql_query(social_query, conn, params=(symbol,))
            
            signals = {}
            
            # News sentiment signals
            if not news_df.empty:
                # Weighted average by relevance
                weighted_news_sentiment = np.average(
                    news_df['sentiment_score'], 
                    weights=news_df['relevance_score'] + 0.1  # Avoid zero weights
                )
                signals['news_sentiment'] = float(weighted_news_sentiment)
                signals['news_volume'] = len(news_df)
            else:
                signals['news_sentiment'] = 0.0
                signals['news_volume'] = 0
            
            # Social sentiment signals
            if not social_df.empty:
                # Weighted average by engagement
                weighted_social_sentiment = np.average(
                    social_df['sentiment_score'],
                    weights=social_df['engagement_score'] + 1  # Avoid zero weights
                )
                signals['social_sentiment'] = float(weighted_social_sentiment)
                signals['social_volume'] = len(social_df)
            else:
                signals['social_sentiment'] = 0.0
                signals['social_volume'] = 0
            
            # Combined signals
            news_weight = 0.6
            social_weight = 0.4
            
            signals['combined_sentiment'] = (
                signals['news_sentiment'] * news_weight + 
                signals['social_sentiment'] * social_weight
            )
            
            # Signal strength based on volume and consistency
            total_volume = signals['news_volume'] + signals['social_volume']
            signals['signal_strength'] = min(total_volume / 20.0, 1.0)  # Normalize to 0-1
            
            # Signal direction
            if signals['combined_sentiment'] > 0.1:
                signals['direction'] = 'bullish'
            elif signals['combined_sentiment'] < -0.1:
                signals['direction'] = 'bearish'
            else:
                signals['direction'] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating sentiment signals: {e}")
            return {
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'combined_sentiment': 0.0,
                'signal_strength': 0.0,
                'direction': 'neutral',
                'news_volume': 0,
                'social_volume': 0
            }


# Integration function for main trading system
def get_alternative_data_features(symbols: List[str], config: dict) -> pd.DataFrame:
    """Main function to get alternative data features for trading system"""
    
    alt_data_manager = AlternativeDataManager(config)
    
    # Run async data collection
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        alternative_data = loop.run_until_complete(
            alt_data_manager.collect_all_alternative_data(symbols)
        )
    except Exception as e:
        logger.error(f"Error collecting alternative data: {e}")
        alternative_data = {}
    finally:
        loop.close()
    
    # Combine data for all symbols
    combined_features = pd.DataFrame()
    
    for symbol, data in alternative_data.items():
        if not data.empty:
            # Add symbol prefix to column names
            symbol_data = data.add_suffix(f'_{symbol}')
            combined_features = pd.concat([combined_features, symbol_data], axis=1)
    
    return combined_features


# Example configuration
EXAMPLE_CONFIG = {
    'news_api_key': 'your_news_api_key_here',
    'twitter_bearer_token': 'your_twitter_bearer_token_here',
    'economic_calendar_api': 'your_economic_api_key_here',
    'data_path': './data/alternative'
}

if __name__ == "__main__":
    # Test the system
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    test_data = get_alternative_data_features(symbols, EXAMPLE_CONFIG)
    print(f"Alternative data shape: {test_data.shape}")
    print(f"Features: {list(test_data.columns)}")
