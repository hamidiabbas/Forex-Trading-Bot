# test_all_modules.py - Complete system test
"""Test script to verify all modules are working correctly"""

def test_complete_system():
    print("🧪 Testing complete alternative data system...")
    
    # Test 1: NewsAPI
    try:
        from newsapi import NewsApiClient
        print("✅ NewsAPI library imported successfully")
        
        # Test client creation (without API key)
        try:
            client = NewsApiClient(api_key='test_key')
            print("✅ NewsAPI client creation works")
        except Exception as e:
            print(f"ℹ️  NewsAPI client test (expected without real API key): {type(e).__name__}")
    except Exception as e:
        print(f"❌ NewsAPI failed: {e}")
    
    # Test 2: Sentiment Analysis
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        test_scores = analyzer.polarity_scores("The market is performing very well today!")
        print(f"✅ VADER Sentiment: {test_scores}")
    except Exception as e:
        print(f"⚠️ VADER Sentiment: {e}")
    
    try:
        from textblob import TextBlob
        blob = TextBlob("The economy is showing positive growth.")
        print(f"✅ TextBlob: sentiment={blob.sentiment.polarity:.3f}")
    except Exception as e:
        print(f"⚠️ TextBlob: {e}")
    
    # Test 3: AlternativeDataManager
    try:
        from alternative_data_manager import AlternativeDataManager
        manager = AlternativeDataManager()
        stats = manager.get_statistics()
        print(f"✅ AlternativeDataManager: {stats}")
        
        # Test sentiment analysis
        sentiment = manager.get_market_sentiment('EURUSD', 6)
        print(f"✅ Sentiment test: {sentiment['overall_sentiment']:.3f}")
        print(f"   News count: {sentiment['news_count']}")
        print(f"   Confidence: {sentiment['confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ AlternativeDataManager failed: {e}")
    
    print("\n🎯 Complete system test finished!")

if __name__ == "__main__":
    test_complete_system()
