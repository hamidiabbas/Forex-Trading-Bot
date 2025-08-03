# advanced_feature_fusion_networks.py - Professional Feature Processing Architectures

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Configuration for advanced neural networks"""
    feature_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = 'gelu'
    use_layer_norm: bool = True
    use_residual: bool = True

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for comprehensive analysis"""[1]
    
    def __init__(self, input_dim: int, config: NetworkConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        
        # Multi-scale convolution layers
        self.scale_1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.scale_2 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.scale_3 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.scale_4 = nn.Conv1d(1, 64, kernel_size=11, padding=5)
        
        # Feature fusion
        self.fusion = nn.Linear(256, config.feature_dim)
        self.layer_norm = nn.LayerNorm(config.feature_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Multi-scale feature extraction
        scale1_out = F.relu(self.scale_1(x))  # (batch, 64, features)
        scale2_out = F.relu(self.scale_2(x))
        scale3_out = F.relu(self.scale_3(x))
        scale4_out = F.relu(self.scale_4(x))
        
        # Global pooling for each scale
        scale1_pool = F.adaptive_avg_pool1d(scale1_out, 1).squeeze(-1)
        scale2_pool = F.adaptive_avg_pool1d(scale2_out, 1).squeeze(-1)
        scale3_pool = F.adaptive_avg_pool1d(scale3_out, 1).squeeze(-1)
        scale4_pool = F.adaptive_avg_pool1d(scale4_out, 1).squeeze(-1)
        
        # Concatenate multi-scale features
        fused = torch.cat([scale1_pool, scale2_pool, scale3_pool, scale4_pool], dim=1)
        
        # Final fusion
        output = self.fusion(fused)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output

class SelfAttentionFeatureProcessor(nn.Module):
    """Self-attention mechanism for feature relationships"""[1]
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.attention_dim = config.feature_dim
        
        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim * 4),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.attention_dim * 4, self.attention_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.attention_dim)
        self.norm2 = nn.LayerNorm(self.attention_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, attention_weights = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class HierarchicalFeatureNetwork(nn.Module):
    """Hierarchical processing of different feature types"""[1]
    
    def __init__(self, feature_groups: Dict[str, int], config: NetworkConfig):
        super().__init__()
        self.feature_groups = feature_groups
        self.config = config
        
        # Separate processors for different feature types
        self.group_processors = nn.ModuleDict()
        for group_name, group_size in feature_groups.items():
            self.group_processors[group_name] = nn.Sequential(
                nn.Linear(group_size, config.feature_dim // 2),
                nn.LayerNorm(config.feature_dim // 2),
                nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.feature_dim // 2, config.feature_dim // 4)
            )
        
        # Cross-group attention
        total_groups = len(feature_groups)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim // 4,
            num_heads=min(config.num_heads, 4),
            dropout=config.dropout,
            batch_first=True
        )
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(total_groups * config.feature_dim // 4, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        group_outputs = []
        
        # Process each feature group
        for group_name, features in feature_dict.items():
            if group_name in self.group_processors:
                processed = self.group_processors[group_name](features)
                group_outputs.append(processed)
        
        if not group_outputs:
            raise ValueError("No valid feature groups found")
        
        # Stack for cross-attention
        stacked_groups = torch.stack(group_outputs, dim=1)  # (batch, groups, features)
        
        # Apply cross-group attention
        attended_groups, _ = self.cross_attention(stacked_groups, stacked_groups, stacked_groups)
        
        # Flatten and fuse
        flattened = attended_groups.flatten(start_dim=1)
        final_output = self.final_fusion(flattened)
        
        return final_output

class AdaptiveFeatureSelector(nn.Module):
    """Adaptive feature selection with learned importance weights"""[1]
    
    def __init__(self, input_dim: int, config: NetworkConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        
        # Feature importance network
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation network
        self.transform_net = nn.Sequential(
            nn.Linear(input_dim, config.feature_dim * 2),
            nn.LayerNorm(config.feature_dim * 2),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim * 2, config.feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate feature importance
        importance_weights = self.importance_net(x)
        
        # Apply adaptive selection
        selected_features = x * importance_weights
        
        # Transform selected features
        transformed = self.transform_net(selected_features)
        
        return transformed, importance_weights

class AdvancedFeatureFusionNetwork(nn.Module):
    """Complete advanced feature fusion network for 100+ features"""[1]
    
    def __init__(self, 
                 input_dim: int,
                 feature_groups: Dict[str, int],
                 output_dim: int,
                 config: NetworkConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Multi-scale feature extraction
        self.multi_scale_extractor = MultiScaleFeatureExtractor(input_dim, config)
        
        # Adaptive feature selection
        self.feature_selector = AdaptiveFeatureSelector(input_dim, config)
        
        # Hierarchical feature processing
        self.hierarchical_processor = HierarchicalFeatureNetwork(feature_groups, config)
        
        # Transformer layers for sequential processing
        self.transformer_layers = nn.ModuleList([
            SelfAttentionFeatureProcessor(config) 
            for _ in range(config.num_layers)
        ])
        
        # Feature fusion and combination
        fusion_dim = config.feature_dim * 3  # multi-scale + selected + hierarchical
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.feature_dim * 2),
            nn.LayerNorm(config.feature_dim * 2),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim * 2, config.feature_dim),
            nn.LayerNorm(config.feature_dim)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                features: torch.Tensor,
                feature_groups: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = features.size(0)
        
        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale_extractor(features)
        
        # Adaptive feature selection
        selected_features, importance_weights = self.feature_selector(features)
        
        # Hierarchical feature processing
        hierarchical_features = self.hierarchical_processor(feature_groups)
        
        # Combine all feature types
        combined_features = torch.cat([
            multi_scale_features,
            selected_features,
            hierarchical_features
        ], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Add sequence dimension for transformer processing
        sequence_features = fused_features.unsqueeze(1)  # (batch, 1, features)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            sequence_features = transformer_layer(sequence_features)
        
        # Remove sequence dimension
        final_features = sequence_features.squeeze(1)
        
        # Generate output
        output = self.output_head(final_features)
        
        return {
            'output': output,
            'features': final_features,
            'importance_weights': importance_weights,
            'multi_scale_features': multi_scale_features,
            'hierarchical_features': hierarchical_features
        }

class EnhancedTradingTransformer(nn.Module):
    """Enhanced Transformer specifically designed for trading features"""[1]
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Positional encoding for temporal features
        self.positional_encoding = PositionalEncoding(config.feature_dim, config.dropout)
        
        # Feature embedding layers
        self.feature_embeddings = nn.ModuleDict({
            'price': nn.Linear(8, config.feature_dim // 4),
            'technical': nn.Linear(16, config.feature_dim // 4),
            'volume': nn.Linear(4, config.feature_dim // 4),
            'sentiment': nn.Linear(4, config.feature_dim // 4),
            'market_structure': nn.Linear(8, config.feature_dim // 4),
            'temporal': nn.Linear(12, config.feature_dim // 4)
        })
        
        # Cross-attention between different feature types
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.feature_dim // 4,
                num_heads=2,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Final transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feature_dim * 4,
            dropout=config.dropout,
            activation='gelu' if config.activation == 'gelu' else 'relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
    
    def forward(self, feature_groups: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_features = []
        
        # Embed each feature group
        for group_name, features in feature_groups.items():
            if group_name in self.feature_embeddings:
                embedded = self.feature_embeddings[group_name](features)
                embedded_features.append(embedded)
        
        # Concatenate all embedded features
        combined_embedded = torch.cat(embedded_features, dim=-1)
        
        # Add positional encoding
        sequence_features = combined_embedded.unsqueeze(1)
        sequence_features = self.positional_encoding(sequence_features)
        
        # Apply transformer encoding
        encoded_features = self.transformer_encoder(sequence_features)
        
        return encoded_features.squeeze(1)

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal features"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Integration Manager for Neural Networks
class NeuralNetworkManager:
    """Manager for advanced neural network architectures"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_feature_fusion_network(self, 
                                    input_dim: int,
                                    feature_groups: Dict[str, int],
                                    output_dim: int = 3) -> AdvancedFeatureFusionNetwork:
        """Create advanced feature fusion network"""
        
        model = AdvancedFeatureFusionNetwork(
            input_dim=input_dim,
            feature_groups=feature_groups,
            output_dim=output_dim,
            config=self.config
        ).to(self.device)
        
        self.models['feature_fusion'] = model
        return model
    
    def create_trading_transformer(self) -> EnhancedTradingTransformer:
        """Create enhanced trading transformer"""
        
        model = EnhancedTradingTransformer(self.config).to(self.device)
        self.models['trading_transformer'] = model
        return model
    
    def get_model_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of all models"""
        summary = {}
        
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            summary[name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            }
        
        return summary
