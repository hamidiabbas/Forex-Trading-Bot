# test_dependencies.py - Complete dependency test
"""Test script to verify all dependencies are working correctly"""

def test_all_dependencies():
    print("🧪 Testing all dependencies...")
    
    # Test 1: VADER Sentiment
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        test_scores = analyzer.polarity_scores("The market is performing very well today!")
        print(f"✅ VADER Sentiment: {test_scores}")
    except Exception as e:
        print(f"❌ VADER Sentiment failed: {e}")
    
    # Test 2: TextBlob
    try:
        from textblob import TextBlob
        blob = TextBlob("The economy is showing positive growth.")
        print(f"✅ TextBlob: sentiment={blob.sentiment.polarity:.3f}")
    except Exception as e:
        print(f"❌ TextBlob failed: {e}")
    
    # Test 3: AlternativeDataManager
    try:
        from alternative_data_manager import AlternativeDataManager
        manager = AlternativeDataManager()
        stats = manager.get_statistics()
        print(f"✅ AlternativeDataManager: {stats}")
        
        # Test sentiment analysis
        sentiment = manager.get_market_sentiment('EURUSD', 6)
        print(f"✅ Sentiment test: {sentiment['overall_sentiment']:.3f}")
        
    except Exception as e:
        print(f"❌ AlternativeDataManager failed: {e}")
    
    # Test 4: Other dependencies
    try:
        import pandas as pd
        import numpy as np
        import requests
        print("✅ Core dependencies (pandas, numpy, requests) working")
    except Exception as e:
        print(f"❌ Core dependencies failed: {e}")
    
    print("\n🎯 Dependency test completed!")

if __name__ == "__main__":
    test_all_dependencies()
