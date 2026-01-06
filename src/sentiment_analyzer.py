"""
Sentiment Analysis Module
Uses FinBERT to analyze financial news sentiment
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Analyze sentiment of financial news using FinBERT"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("FinBERT loaded successfully!")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment_score': 0.0
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        probs = probs.cpu().numpy()[0]
        
        # FinBERT outputs: [positive, negative, neutral]
        return {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'sentiment_score': float(probs[0] - probs[1])  # Range: -1 to +1
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze multiple texts in batches for efficiency
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter out None/NaN
            valid_batch = [str(t) if t and not pd.isna(t) else "" for t in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                valid_batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs.cpu().numpy()
            
            # Process each result
            for prob in probs:
                results.append({
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2]),
                    'sentiment_score': float(prob[0] - prob[1])
                })
            
            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return results
    
    def analyze_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for all news in a DataFrame
        
        Args:
            df: DataFrame with 'title' and optionally 'description' columns
            
        Returns:
            DataFrame with added sentiment columns
        """
        print(f"Analyzing sentiment for {len(df)} articles...")
        
        # Combine title and description for better context
        if 'description' in df.columns:
            texts = (df['title'].fillna('') + '. ' + df['description'].fillna('')).tolist()
        else:
            texts = df['title'].fillna('').tolist()
        
        # Analyze
        sentiments = self.analyze_batch(texts)
        
        # Add to dataframe
        sentiment_df = pd.DataFrame(sentiments)
        result = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        print("Sentiment analysis complete!")
        return result
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news sentiment by day
        
        Args:
            df: DataFrame with 'date' and sentiment columns
            
        Returns:
            Daily aggregated sentiment DataFrame
        """
        df['date'] = pd.to_datetime(df['date'])
        
        daily = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        })
        
        daily.columns = [
            'sentiment_mean',
            'sentiment_std',
            'news_count',
            'positive_ratio',
            'negative_ratio',
            'neutral_ratio'
        ]
        
        # Fill std with 0 for single-article days
        daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
        
        # Calculate sentiment intensity (absolute value)
        daily['sentiment_intensity'] = daily['sentiment_mean'].abs()
        
        # News momentum (rolling average)
        daily['sentiment_ma_3'] = daily['sentiment_mean'].rolling(3, min_periods=1).mean()
        daily['sentiment_ma_7'] = daily['sentiment_mean'].rolling(7, min_periods=1).mean()
        
        return daily.reset_index()


def process_news_sentiment(news_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Complete pipeline: Load news CSV, analyze sentiment, aggregate daily
    
    Args:
        news_csv: Path to news CSV file
        output_csv: Optional output path
        
    Returns:
        Daily sentiment DataFrame
    """
    # Load news
    df = pd.read_csv(news_csv)
    print(f"Loaded {len(df)} articles from {news_csv}")
    
    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    df_with_sentiment = analyzer.analyze_news_dataframe(df)
    
    # Aggregate daily
    daily_sentiment = analyzer.aggregate_daily_sentiment(df_with_sentiment)
    
    # Save if requested
    if output_csv:
        daily_sentiment.to_csv(output_csv, index=False)
        print(f"Saved daily sentiment to {output_csv}")
    
    return daily_sentiment


if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test single text
    test_texts = [
        "Apple reports record quarterly earnings, stock surges 10%",
        "Company faces major lawsuit, shares plummet",
        "Trading continues normally with no major news"
    ]
    
    print("\nTesting individual texts:")
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Score: {result['sentiment_score']:.3f} (Pos: {result['positive']:.2f}, Neg: {result['negative']:.2f})")
