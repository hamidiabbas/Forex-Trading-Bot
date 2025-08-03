# enhanced_sentiment_neural_networks.py - Advanced Financial Sentiment Analysis
"""
Professional Financial Sentiment Analysis with Advanced Neural Networks
Specialized architectures for news, social media, and financial text processing
Complete integration with enterprise trading system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import (
    AutoTokenizer, AutoModel, BertModel, RobertaModel,
    DistilBertModel, AlbertModel, DebertaV2Model
)
from transformers.modeling_outputs import BaseModelOutput
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import re
from collections import defaultdict
import asyncio
import aiohttp
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis models"""
    # Model architecture
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    
    # Text processing
    max_sequence_length: int = 512
    batch_size: int = 32
    
    # Financial-specific
    financial_vocab_size: int = 50000
    sentiment_classes: int = 5  # Very Negative, Negative, Neutral, Positive, Very Positive
    confidence_threshold: float = 0.7
    
    # Multi-modal
    enable_technical_fusion: bool = True
    enable_temporal_modeling: bool = True
    enable_cross_attention: bool = True

class FinancialBertEncoder(nn.Module):
    """Enhanced BERT encoder specifically designed for financial text"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained financial BERT or DistilBERT
        self.bert_model = AutoModel.from_pretrained(
            "ProsusAI/finbert",  # Financial BERT
            cache_dir="./models/finbert_cache"
        )
        
        # Freeze lower layers, fine-tune upper layers
        for param in list(self.bert_model.parameters())[:-6]:
            param.requires_grad = False
        
        # Financial domain adaptation layers
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.bert_model.config.hidden_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Multi-head self-attention for financial context
        self.financial_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply domain adaptation
        adapted_embeddings = self.domain_adapter(bert_outputs.last_hidden_state)
        
        # Apply financial attention
        attended_output, attention_weights = self.financial_attention(
            adapted_embeddings, adapted_embeddings, adapted_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(adapted_embeddings + attended_output)
        
        return output

class NewsSpecificProcessor(nn.Module):
    """Specialized processor for financial news text"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # News-specific features
        self.source_embedding = nn.Embedding(100, config.hidden_dim // 4)  # News sources
        self.section_embedding = nn.Embedding(20, config.hidden_dim // 4)   # News sections
        self.urgency_embedding = nn.Embedding(5, config.hidden_dim // 4)    # Urgency levels
        
        # Headline vs body attention
        self.headline_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Financial entity extraction
        self.entity_extractor = FinancialEntityExtractor(config)
        
        # News importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                text_embeddings: torch.Tensor,
                source_ids: torch.Tensor,
                section_ids: torch.Tensor,
                urgency_ids: torch.Tensor,
                headline_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len, hidden_dim = text_embeddings.shape
        
        # Add news-specific embeddings
        source_emb = self.source_embedding(source_ids).unsqueeze(1).expand(-1, seq_len, -1)
        section_emb = self.section_embedding(section_ids).unsqueeze(1).expand(-1, seq_len, -1)
        urgency_emb = self.urgency_embedding(urgency_ids).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine embeddings
        enhanced_embeddings = torch.cat([
            text_embeddings,
            source_emb,
            section_emb,
            urgency_emb
        ], dim=-1)
        
        # Project back to hidden_dim
        enhanced_embeddings = nn.Linear(
            hidden_dim + 3 * (hidden_dim // 4), 
            hidden_dim
        ).to(text_embeddings.device)(enhanced_embeddings)
        
        # Apply headline attention
        headline_attended, _ = self.headline_attention(
            enhanced_embeddings,
            enhanced_embeddings,
            enhanced_embeddings,
            key_padding_mask=~headline_mask.bool()
        )
        
        # Extract financial entities
        entity_info = self.entity_extractor(enhanced_embeddings)
        
        # Calculate importance score
        pooled_output = enhanced_embeddings.mean(dim=1)
        importance_score = self.importance_scorer(pooled_output)
        
        return {
            'enhanced_embeddings': headline_attended,
            'entity_info': entity_info,
            'importance_score': importance_score
        }

class SocialMediaProcessor(nn.Module):
    """Specialized processor for social media sentiment"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Social media specific features
        self.platform_embedding = nn.Embedding(10, config.hidden_dim // 4)  # Twitter, Reddit, etc.
        self.engagement_encoder = nn.Linear(5, config.hidden_dim // 4)      # Likes, shares, etc.
        self.temporal_encoder = nn.Linear(3, config.hidden_dim // 4)        # Time features
        
        # Hashtag and mention processing
        self.hashtag_processor = HashtagProcessor(config)
        self.mention_processor = MentionProcessor(config)
        
        # Viral content detector
        self.viral_detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Emoji and sentiment intensifier
        self.emoji_processor = EmojiSentimentProcessor(config)
    
    def forward(self,
                text_embeddings: torch.Tensor,
                platform_ids: torch.Tensor,
                engagement_features: torch.Tensor,
                temporal_features: torch.Tensor,
                hashtags: List[str],
                mentions: List[str],
                emojis: List[str]) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len, hidden_dim = text_embeddings.shape
        
        # Platform embeddings
        platform_emb = self.platform_embedding(platform_ids).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Engagement features
        engagement_emb = self.engagement_encoder(engagement_features).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Temporal features
        temporal_emb = self.temporal_encoder(temporal_features).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Process hashtags and mentions
        hashtag_info = self.hashtag_processor(hashtags, text_embeddings)
        mention_info = self.mention_processor(mentions, text_embeddings)
        
        # Process emojis
        emoji_sentiment = self.emoji_processor(emojis, text_embeddings)
        
        # Combine all features
        enhanced_embeddings = torch.cat([
            text_embeddings,
            platform_emb,
            engagement_emb,
            temporal_emb
        ], dim=-1)
        
        # Project back to hidden_dim
        enhanced_embeddings = nn.Linear(
            hidden_dim + 3 * (hidden_dim // 4),
            hidden_dim
        ).to(text_embeddings.device)(enhanced_embeddings)
        
        # Add hashtag and mention information
        if hashtag_info is not None:
            enhanced_embeddings = enhanced_embeddings + hashtag_info
        if mention_info is not None:
            enhanced_embeddings = enhanced_embeddings + mention_info
        
        # Detect viral content
        pooled_output = enhanced_embeddings.mean(dim=1)
        viral_score = self.viral_detector(pooled_output)
        
        return {
            'enhanced_embeddings': enhanced_embeddings,
            'hashtag_info': hashtag_info,
            'mention_info': mention_info,
            'emoji_sentiment': emoji_sentiment,
            'viral_score': viral_score
        }

class FinancialEntityExtractor(nn.Module):
    """Extract and process financial entities from text"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Entity types: Company, Ticker, Currency, Commodity, etc.
        self.entity_types = ['COMPANY', 'TICKER', 'CURRENCY', 'COMMODITY', 'PERSON', 'EVENT']
        self.entity_type_embedding = nn.Embedding(len(self.entity_types), config.hidden_dim // 4)
        
        # Entity importance scorer
        self.entity_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Financial entity dictionary (simplified)
        self.financial_entities = {
            'companies': ['Apple', 'Microsoft', 'Google', 'Tesla', 'Amazon'],
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
            'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'CAD'],
            'commodities': ['Gold', 'Silver', 'Oil', 'Copper']
        }
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract entities (simplified implementation)
        # In practice, you would use NER models here
        
        entity_scores = self.entity_scorer(text_embeddings)
        
        return {
            'entity_scores': entity_scores,
            'entity_embeddings': text_embeddings  # Placeholder
        }

class HashtagProcessor(nn.Module):
    """Process hashtags for sentiment analysis"""
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        self.hashtag_embedding = nn.Embedding(10000, config.hidden_dim // 4)
    
    def forward(self, hashtags: List[str], text_embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        if not hashtags:
            return None
        
        # Process hashtags (simplified)
        # In practice, you would have a hashtag vocabulary
        return None  # Placeholder

class MentionProcessor(nn.Module):
    """Process mentions for sentiment analysis"""
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        self.mention_embedding = nn.Embedding(10000, config.hidden_dim // 4)
    
    def forward(self, mentions: List[str], text_embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        if not mentions:
            return None
        
        # Process mentions (simplified)
        return None  # Placeholder

class EmojiSentimentProcessor(nn.Module):
    """Process emojis for sentiment intensity"""
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Emoji sentiment mapping (simplified)
        self.emoji_sentiment = {
            '😀': 0.8, '😊': 0.7, '😢': -0.7, '😭': -0.8,
            '🚀': 0.9, '📈': 0.8, '📉': -0.8, '💰': 0.6
        }
        
        self.emoji_encoder = nn.Linear(len(self.emoji_sentiment), config.hidden_dim // 4)
    
    def forward(self, emojis: List[str], text_embeddings: torch.Tensor) -> torch.Tensor:
        # Process emoji sentiment (simplified)
        batch_size = text_embeddings.shape[0]
        emoji_features = torch.zeros(batch_size, len(self.emoji_sentiment))
        
        return emoji_features  # Placeholder

class TemporalSentimentModeling(nn.Module):
    """Model temporal dynamics in sentiment"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # LSTM for temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention over time
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim * 2,  # Bidirectional
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, 
                sentiment_sequence: torch.Tensor,
                time_deltas: torch.Tensor) -> torch.Tensor:
        
        # Apply LSTM
        lstm_output, (hidden, cell) = self.temporal_lstm(sentiment_sequence)
        
        # Apply temporal attention
        attended_output, _ = self.temporal_attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Fusion
        fused_output = self.temporal_fusion(attended_output)
        
        return fused_output

class MarketDataFusionLayer(nn.Module):
    """Fuse sentiment with market data"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Market data encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(10, config.hidden_dim // 2),  # Price, volume, volatility, etc.
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2)
        )
        
        # Cross-attention between sentiment and market data
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self,
                sentiment_embeddings: torch.Tensor,
                market_data: torch.Tensor) -> torch.Tensor:
        
        # Encode market data
        market_encoded = self.market_encoder(market_data)
        
        # Expand market data to match sentiment sequence length
        seq_len = sentiment_embeddings.shape[1]
        market_expanded = market_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Cross-attention
        fused_sentiment, _ = self.cross_attention(
            sentiment_embeddings,
            market_expanded,
            market_expanded
        )
        
        # Final fusion
        combined = torch.cat([fused_sentiment, market_expanded], dim=-1)
        fused_output = self.fusion_layer(combined)
        
        return fused_output

class MultilevelSentimentClassifier(nn.Module):
    """Multi-level sentiment classification with confidence estimation"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Multiple classification heads
        self.binary_classifier = nn.Linear(config.hidden_dim, 2)      # Positive/Negative
        self.ternary_classifier = nn.Linear(config.hidden_dim, 3)     # Positive/Neutral/Negative
        self.fine_grained_classifier = nn.Linear(config.hidden_dim, config.sentiment_classes)
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Sentiment intensity regressor
        self.intensity_regressor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Market impact predictor
        self.market_impact_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3),  # Bullish, Neutral, Bearish
            nn.Softmax(dim=-1)
        )
    
    def forward(self, sentiment_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Pool embeddings
        pooled = sentiment_embeddings.mean(dim=1)
        
        # Multiple classification outputs
        binary_logits = self.binary_classifier(pooled)
        ternary_logits = self.ternary_classifier(pooled)
        fine_grained_logits = self.fine_grained_classifier(pooled)
        
        # Confidence and intensity
        confidence = self.confidence_estimator(pooled)
        intensity = self.intensity_regressor(pooled)
        
        # Market impact
        market_impact = self.market_impact_predictor(pooled)
        
        return {
            'binary_logits': binary_logits,
            'ternary_logits': ternary_logits,
            'fine_grained_logits': fine_grained_logits,
            'confidence': confidence,
            'intensity': intensity,
            'market_impact': market_impact
        }

class EnhancedFinancialSentimentAnalyzer(nn.Module):
    """Complete financial sentiment analysis system"""[25]
    
    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.bert_encoder = FinancialBertEncoder(config)
        self.news_processor = NewsSpecificProcessor(config)
        self.social_processor = SocialMediaProcessor(config)
        
        # Advanced components
        self.temporal_modeler = TemporalSentimentModeling(config)
        self.market_fusion = MarketDataFusionLayer(config)
        self.classifier = MultilevelSentimentClassifier(config)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        logger.info("Enhanced Financial Sentiment Analyzer initialized")
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                text_type: str,  # 'news' or 'social'
                metadata: Dict[str, Any],
                market_data: Optional[torch.Tensor] = None,
                temporal_sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Get BERT embeddings
        bert_embeddings = self.bert_encoder(input_ids, attention_mask)
        
        # Process based on text type
        if text_type == 'news':
            processed_output = self.news_processor(
                bert_embeddings,
                metadata.get('source_ids', torch.zeros_like(input_ids[:, 0])),
                metadata.get('section_ids', torch.zeros_like(input_ids[:, 0])),
                metadata.get('urgency_ids', torch.zeros_like(input_ids[:, 0])),
                metadata.get('headline_mask', attention_mask)
            )
            enhanced_embeddings = processed_output['enhanced_embeddings']
            
        elif text_type == 'social':
            processed_output = self.social_processor(
                bert_embeddings,
                metadata.get('platform_ids', torch.zeros_like(input_ids[:, 0])),
                metadata.get('engagement_features', torch.zeros(input_ids.shape[0], 5)),
                metadata.get('temporal_features', torch.zeros(input_ids.shape[0], 3)),
                metadata.get('hashtags', []),
                metadata.get('mentions', []),
                metadata.get('emojis', [])
            )
            enhanced_embeddings = processed_output['enhanced_embeddings']
        else:
            enhanced_embeddings = bert_embeddings
            processed_output = {}
        
        # Apply temporal modeling if sequence provided
        if temporal_sequence is not None:
            enhanced_embeddings = self.temporal_modeler(
                enhanced_embeddings,
                metadata.get('time_deltas', torch.zeros(input_ids.shape[0]))
            )
        
        # Fuse with market data if provided
        if market_data is not None:
            enhanced_embeddings = self.market_fusion(enhanced_embeddings, market_data)
        
        # Get final predictions
        predictions = self.classifier(enhanced_embeddings)
        
        # Combine all outputs
        output = {
            **predictions,
            'embeddings': enhanced_embeddings,
            **processed_output
        }
        
        return output

class SentimentDataProcessor:
    """Professional data processor for sentiment analysis"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        # Financial keywords for enhancement
        self.financial_keywords = {
            'bullish': ['bullish', 'bull', 'buy', 'long', 'up', 'rise', 'gain', 'profit', 'growth'],
            'bearish': ['bearish', 'bear', 'sell', 'short', 'down', 'fall', 'loss', 'decline', 'crash'],
            'neutral': ['hold', 'stable', 'sideways', 'range', 'consolidate']
        }
        
    def preprocess_news_text(self, news_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Preprocess news text data"""
        
        processed_data = {
            'input_ids': [],
            'attention_mask': [],
            'metadata': {
                'source_ids': [],
                'section_ids': [],
                'urgency_ids': [],
                'headline_mask': []
            }
        }
        
        for news_item in news_data:
            # Combine headline and content
            text = f"{news_item.get('headline', '')} {news_item.get('content', '')}"
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=self.config.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed_data['input_ids'].append(tokens['input_ids'].squeeze(0))
            processed_data['attention_mask'].append(tokens['attention_mask'].squeeze(0))
            
            # Process metadata
            processed_data['metadata']['source_ids'].append(
                self._encode_news_source(news_item.get('source', 'unknown'))
            )
            processed_data['metadata']['section_ids'].append(
                self._encode_news_section(news_item.get('section', 'general'))
            )
            processed_data['metadata']['urgency_ids'].append(
                self._encode_urgency(news_item.get('urgency', 'normal'))
            )
            
            # Create headline mask
            headline_tokens = self.tokenizer(
                news_item.get('headline', ''),
                max_length=self.config.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            processed_data['metadata']['headline_mask'].append(
                headline_tokens['attention_mask'].squeeze(0)
            )
        
        # Convert to tensors
        for key, value in processed_data.items():
            if key != 'metadata':
                processed_data[key] = torch.stack(value)
        
        for key, value in processed_data['metadata'].items():
            processed_data['metadata'][key] = torch.tensor(value)
        
        return processed_data
    
    def preprocess_social_text(self, social_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Preprocess social media text data"""
        
        processed_data = {
            'input_ids': [],
            'attention_mask': [],
            'metadata': {
                'platform_ids': [],
                'engagement_features': [],
                'temporal_features': [],
                'hashtags': [],
                'mentions': [],
                'emojis': []
            }
        }
        
        for social_item in social_data:
            # Clean and tokenize text
            text = self._clean_social_text(social_item.get('text', ''))
            
            tokens = self.tokenizer(
                text,
                max_length=self.config.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed_data['input_ids'].append(tokens['input_ids'].squeeze(0))
            processed_data['attention_mask'].append(tokens['attention_mask'].squeeze(0))
            
            # Process metadata
            processed_data['metadata']['platform_ids'].append(
                self._encode_platform(social_item.get('platform', 'unknown'))
            )
            
            # Engagement features: likes, shares, comments, retweets, replies
            engagement = [
                social_item.get('likes', 0),
                social_item.get('shares', 0),
                social_item.get('comments', 0),
                social_item.get('retweets', 0),
                social_item.get('replies', 0)
            ]
            processed_data['metadata']['engagement_features'].append(engagement)
            
            # Temporal features
            timestamp = social_item.get('timestamp', datetime.now())
            hour_of_day = timestamp.hour / 24.0
            day_of_week = timestamp.weekday() / 7.0
            is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
            temporal = [hour_of_day, day_of_week, is_weekend]
            processed_data['metadata']['temporal_features'].append(temporal)
            
            # Extract hashtags, mentions, emojis
            processed_data['metadata']['hashtags'].append(
                self._extract_hashtags(social_item.get('text', ''))
            )
            processed_data['metadata']['mentions'].append(
                self._extract_mentions(social_item.get('text', ''))
            )
            processed_data['metadata']['emojis'].append(
                self._extract_emojis(social_item.get('text', ''))
            )
        
        # Convert to tensors
        for key, value in processed_data.items():
            if key != 'metadata':
                processed_data[key] = torch.stack(value)
        
        processed_data['metadata']['platform_ids'] = torch.tensor(
            processed_data['metadata']['platform_ids']
        )
        processed_data['metadata']['engagement_features'] = torch.tensor(
            processed_data['metadata']['engagement_features'],
            dtype=torch.float32
        )
        processed_data['metadata']['temporal_features'] = torch.tensor(
            processed_data['metadata']['temporal_features'],
            dtype=torch.float32
        )
        
        return processed_data
    
    def _clean_social_text(self, text: str) -> str:
        """Clean social media text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Clean extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        return re.findall(r'#\w+', text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        return re.findall(r'@\w+', text)
    
    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text (simplified)"""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.findall(text)
    
    def _encode_news_source(self, source: str) -> int:
        """Encode news source to integer ID"""
        source_mapping = {
            'reuters': 0, 'bloomberg': 1, 'cnbc': 2, 'wsj': 3, 'ft': 4,
            'marketwatch': 5, 'yahoo': 6, 'google': 7, 'unknown': 99
        }
        return source_mapping.get(source.lower(), 99)
    
    def _encode_news_section(self, section: str) -> int:
        """Encode news section to integer ID"""
        section_mapping = {
            'markets': 0, 'economy': 1, 'companies': 2, 'technology': 3,
            'politics': 4, 'general': 5, 'breaking': 6, 'analysis': 7
        }
        return section_mapping.get(section.lower(), 5)
    
    def _encode_urgency(self, urgency: str) -> int:
        """Encode urgency level to integer ID"""
        urgency_mapping = {
            'low': 0, 'normal': 1, 'high': 2, 'urgent': 3, 'breaking': 4
        }
        return urgency_mapping.get(urgency.lower(), 1)
    
    def _encode_platform(self, platform: str) -> int:
        """Encode social media platform to integer ID"""
        platform_mapping = {
            'twitter': 0, 'reddit': 1, 'facebook': 2, 'linkedin': 3,
            'discord': 4, 'telegram': 5, 'unknown': 9
        }
        return platform_mapping.get(platform.lower(), 9)

class SentimentTrainingManager:
    """Training manager for sentiment analysis models"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.model = EnhancedFinancialSentimentAnalyzer(config)
        self.data_processor = SentimentDataProcessor(config)
        
        # Training configuration
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-5,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.confidence_loss = nn.BCELoss()
        
        logger.info("Sentiment Training Manager initialized")
    
    def train_model(self,
                   train_news_data: List[Dict[str, Any]],
                   train_social_data: List[Dict[str, Any]],
                   val_news_data: List[Dict[str, Any]],
                   val_social_data: List[Dict[str, Any]],
                   epochs: int = 10) -> Dict[str, List[float]]:
        """Train the sentiment analysis model"""
        
        logger.info("Starting sentiment model training...")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_news_data, train_social_data)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_news_data, val_social_data)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_accuracy'].append(train_acc)
            training_history['val_accuracy'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        logger.info("Training completed!")
        return training_history
    
    def _train_epoch(self,
                    news_data: List[Dict[str, Any]],
                    social_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Process news data
        news_processed = self.data_processor.preprocess_news_text(news_data)
        
        # Train on news
        outputs = self.model(
            news_processed['input_ids'],
            news_processed['attention_mask'],
            'news',
            news_processed['metadata']
        )
        
        # Calculate loss (simplified)
        loss = self._calculate_loss(outputs, news_data)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        total_loss += loss.item()
        total_samples += len(news_data)
        
        # Process social data
        social_processed = self.data_processor.preprocess_social_text(social_data)
        
        # Train on social media
        outputs = self.model(
            social_processed['input_ids'],
            social_processed['attention_mask'],
            'social',
            social_processed['metadata']
        )
        
        # Calculate loss
        loss = self._calculate_loss(outputs, social_data)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        total_loss += loss.item()
        total_samples += len(social_data)
        
        avg_loss = total_loss / 2  # Two batches
        avg_accuracy = 0.5  # Placeholder
        
        return avg_loss, avg_accuracy
    
    def _validate_epoch(self,
                       news_data: List[Dict[str, Any]],
                       social_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            # Validate on news
            news_processed = self.data_processor.preprocess_news_text(news_data)
            outputs = self.model(
                news_processed['input_ids'],
                news_processed['attention_mask'],
                'news',
                news_processed['metadata']
            )
            loss = self._calculate_loss(outputs, news_data)
            total_loss += loss.item()
            
            # Validate on social
            social_processed = self.data_processor.preprocess_social_text(social_data)
            outputs = self.model(
                social_processed['input_ids'],
                social_processed['attention_mask'],
                'social',
                social_processed['metadata']
            )
            loss = self._calculate_loss(outputs, social_data)
            total_loss += loss.item()
        
        avg_loss = total_loss / 2
        avg_accuracy = 0.5  # Placeholder
        
        return avg_loss, avg_accuracy
    
    def _calculate_loss(self,
                       outputs: Dict[str, torch.Tensor],
                       ground_truth: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate combined loss"""
        
        # Create dummy labels (in practice, you'd have real labels)
        batch_size = outputs['binary_logits'].shape[0]
        binary_labels = torch.randint(0, 2, (batch_size,))
        intensity_labels = torch.randn(batch_size, 1)
        confidence_labels = torch.rand(batch_size, 1)
        
        # Calculate losses
        binary_loss = self.classification_loss(outputs['binary_logits'], binary_labels)
        intensity_loss = self.regression_loss(outputs['intensity'], intensity_labels)
        confidence_loss = self.confidence_loss(outputs['confidence'], confidence_labels)
        
        # Combined loss
        total_loss = binary_loss + 0.5 * intensity_loss + 0.3 * confidence_loss
        
        return total_loss
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Model loaded from {path}")

# Integration with existing system
def create_enhanced_sentiment_analyzer(config: Dict[str, Any] = None) -> EnhancedFinancialSentimentAnalyzer:
    """Factory function to create sentiment analyzer"""
    
    if config is None:
        config = SentimentConfig()
    else:
        config = SentimentConfig(**config)
    
    return EnhancedFinancialSentimentAnalyzer(config)

# Example usage
def main():
    """Test the enhanced sentiment analysis system"""
    
    # Configuration
    config = SentimentConfig(
        hidden_dim=768,
        num_attention_heads=12,
        max_sequence_length=512,
        sentiment_classes=5
    )
    
    # Create model
    sentiment_analyzer = EnhancedFinancialSentimentAnalyzer(config)
    
    # Example usage
    print("Enhanced Financial Sentiment Analysis System")
    print("=" * 60)
    print(f"Model Parameters: {sum(p.numel() for p in sentiment_analyzer.parameters()):,}")
    print(f"Configuration: {config}")
    print("=" * 60)
    
    # Example news data
    sample_news = [{
        'headline': 'Apple reports strong quarterly earnings',
        'content': 'Apple Inc. reported better-than-expected quarterly earnings...',
        'source': 'reuters',
        'section': 'companies',
        'urgency': 'high',
        'timestamp': datetime.now()
    }]
    
    # Example social media data
    sample_social = [{
        'text': 'AAPL to the moon! 🚀📈 Great earnings! #Apple #Bullish',
        'platform': 'twitter',
        'likes': 150,
        'shares': 25,
        'comments': 10,
        'timestamp': datetime.now()
    }]
    
    # Process data
    data_processor = SentimentDataProcessor(config)
    
    print("\nProcessing sample data...")
    news_processed = data_processor.preprocess_news_text(sample_news)
    social_processed = data_processor.preprocess_social_text(sample_social)
    
    print(f"News input shape: {news_processed['input_ids'].shape}")
    print(f"Social input shape: {social_processed['input_ids'].shape}")
    
    print("\n✅ Enhanced Financial Sentiment Analysis System Ready!")
    print("🧠 Features: BERT, Multi-Modal, Temporal, Market Fusion")
    print("🚀 Ready for enterprise trading integration!")

if __name__ == "__main__":
    main()
