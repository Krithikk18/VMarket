"""
News Data Fetcher Module
Fetches news articles from various APIs for sentiment analysis
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os
from dotenv import load_dotenv


class NewsFetcher:
    """Fetch news articles from NewsAPI and other sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news fetcher
        
        Args:
            api_key: NewsAPI key (if None, reads from .env)
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('NEWS_API_KEY')
        
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        
        # Company name mappings for better search
        self.company_keywords = {
            'AAPL': ['Apple', 'iPhone', 'Tim Cook', 'AAPL'],
            'NVDA': ['NVIDIA', 'Jensen Huang', 'NVDA', 'GPU'],
            'MSFT': ['Microsoft', 'Satya Nadella', 'MSFT', 'Azure'],
            'GOOGL': ['Google', 'Alphabet', 'GOOGL', 'Sundar Pichai'],
            'META': ['Meta', 'Facebook', 'Mark Zuckerberg', 'META', 'Instagram'],
            'AMZN': ['Amazon', 'Jeff Bezos', 'Andy Jassy', 'AMZN', 'AWS'],
            'NFLX': ['Netflix', 'NFLX', 'Reed Hastings']
        }
    
    def fetch_news_for_date_range(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        max_articles: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles for a specific date range
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_articles: Maximum articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            print("WARNING: No API key found. Using dummy data.")
            return self._generate_dummy_news(ticker, start_date, end_date)
        
        # Build search query
        keywords = self.company_keywords.get(ticker, [ticker])
        query = ' OR '.join([f'"{kw}"' for kw in keywords[:3]])  # Use top 3 keywords
        
        params = {
            'q': query,
            'from': start_date,
            'to': end_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': min(max_articles, 100),  # API limit
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'date': article.get('publishedAt', '')[:10],  # YYYY-MM-DD
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', '')
                })
            
            return articles
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def fetch_news_batch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        batch_days: int = 7
    ) -> pd.DataFrame:
        """
        Fetch news in batches to avoid API limits
        
        Args:
            ticker: Stock ticker
            start_date: Start date string
            end_date: End date string
            batch_days: Days per batch
            
        Returns:
            DataFrame with all fetched articles
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_articles = []
        current = start
        
        print(f"Fetching news for {ticker} from {start_date} to {end_date}")
        
        while current < end:
            batch_end = min(current + timedelta(days=batch_days), end)
            
            articles = self.fetch_news_for_date_range(
                ticker,
                current.strftime('%Y-%m-%d'),
                batch_end.strftime('%Y-%m-%d')
            )
            
            all_articles.extend(articles)
            
            print(f"  Fetched {len(articles)} articles for {current.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
            
            current = batch_end + timedelta(days=1)
            time.sleep(1)  # Rate limiting
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        else:
            return pd.DataFrame(columns=['date', 'title', 'description', 'source', 'url'])
    
    def _generate_dummy_news(self, ticker: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Generate dummy news for testing when API key is not available
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dummy articles
        """
        templates = [
            f"{ticker} announces strong quarterly earnings",
            f"{ticker} stock rises on positive outlook",
            f"{ticker} faces regulatory challenges",
            f"{ticker} launches new product line",
            f"Analysts upgrade {ticker} rating",
            f"{ticker} CEO discusses future strategy",
            f"Market volatility affects {ticker}",
            f"{ticker} reports record revenue"
        ]
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = (end - start).days
        
        articles = []
        for i in range(min(days, 50)):  # Generate up to 50 dummy articles
            date = start + timedelta(days=i)
            articles.append({
                'date': date.strftime('%Y-%m-%d'),
                'title': templates[i % len(templates)],
                'description': f"News about {ticker}",
                'source': 'DummyNews',
                'url': 'http://example.com'
            })
        
        return articles


def save_news_data(ticker: str, start_date: str, end_date: str, output_dir: str = '.'):
    """
    Fetch and save news data to CSV
    
    Args:
        ticker: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
    """
    fetcher = NewsFetcher()
    df = fetcher.fetch_news_batch(ticker, start_date, end_date)
    
    if not df.empty:
        output_file = os.path.join(output_dir, f"{ticker}_news.csv")
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} articles to {output_file}")
        return df
    else:
        print(f"\nNo articles fetched for {ticker}")
        return None


if __name__ == "__main__":
    # Test news fetcher
    test_ticker = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Testing news fetcher for {test_ticker}")
    print(f"Date range: {start_date} to {end_date}")
    
    df = save_news_data(test_ticker, start_date, end_date)
    if df is not None:
        print(f"\nSample articles:")
        print(df[['date', 'title']].head(10))
