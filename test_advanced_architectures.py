# test_advanced_architectures.py - Testing and configuration

def main():
    """Test advanced neural network architectures"""
    
    # Configuration
    config = NetworkConfig(
        feature_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        activation='gelu'
    )
    
    # Feature groups (based on your 100+ features)
    feature_groups = {
        'price_features': 8,           # OHLC, returns, ratios
        'technical_indicators': 16,    # RSI, MACD, Bollinger, etc.
        'volume_features': 4,          # Volume analysis
        'volatility_features': 8,      # Different volatility measures
        'fractal_features': 4,         # Hurst, fractal dimension
        'entropy_features': 4,         # Shannon, approximate entropy
        'wavelet_features': 8,         # Wavelet decomposition
        'microstructure_features': 8,  # Bid-ask, liquidity
        'regime_features': 4,          # Market regime indicators
        'temporal_features': 12,       # Time-based features
        'sentiment_features': 4,       # News, social sentiment
        'alternative_data': 8,         # Economic calendar, etc.
        'interaction_features': 12     # Feature interactions
    }
    
    total_features = sum(feature_groups.values())  # Should be 100+
    print(f"Total features: {total_features}")
    
    # Create network manager
    network_manager = NeuralNetworkManager(config)
    
    # Create feature fusion network
    fusion_model = network_manager.create_feature_fusion_network(
        input_dim=total_features,
        feature_groups=feature_groups,
        output_dim=3  # BUY, SELL, HOLD
    )
    
    # Create trading transformer
    transformer_model = network_manager.create_trading_transformer()
    
    # Test with sample data
    batch_size = 32
    sample_features = torch.randn(batch_size, total_features)
    
    # Separate into feature groups for testing
    sample_feature_groups = {}
    start_idx = 0
    for group_name, group_size in feature_groups.items():
        end_idx = start_idx + group_size
        sample_feature_groups[group_name] = sample_features[:, start_idx:end_idx]
        start_idx = end_idx
    
    # Test fusion network
    print("Testing Feature Fusion Network...")
    fusion_output = fusion_model(sample_features, sample_feature_groups)
    print(f"Fusion output shape: {fusion_output['output'].shape}")
    print(f"Feature importance shape: {fusion_output['importance_weights'].shape}")
    
    # Test transformer
    print("\nTesting Trading Transformer...")
    transformer_output = transformer_model(sample_feature_groups)
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Get model summary
    print("\nModel Summary:")
    summary = network_manager.get_model_summary()
    for model_name, stats in summary.items():
        print(f"{model_name}:")
        print(f"  Total parameters: {stats['total_parameters']:,}")
        print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
        print(f"  Model size: {stats['model_size_mb']:.2f} MB")
    
    print("\n✅ Advanced neural network architectures ready!")
    print("🧠 Features: Multi-scale extraction, Self-attention, Hierarchical processing")
    print("🚀 Ready for integration with your 100+ advanced features!")

if __name__ == "__main__":
    main()
