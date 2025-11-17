"""
Prediction script for ST-GCN model on ASL Citizen dataset.

Usage:
    python predict_stgcn.py --data path/to/preprocessed.npy
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import model components
from models.stgcn_tf import STGCN, FC, Network


def load_trained_model(weights_path, num_classes=10, dropout=0.05):
    """
    Load trained ST-GCN model by reconstructing architecture and loading weights.
    
    Args:
        weights_path (str): Path to weights file (.weights.h5)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate (must match training)
    
    Returns:
        model: Loaded Network model ready for inference
    """
    print(f"Loading model from: {weights_path}")
    
    # Define skeleton structure (same as training)
    num_nodes = 27
    center = 0
    inward_edges = [
        [1, 0], [2, 0], [3, 1], [4, 2], [5, 3], [6, 4],
        [7, 1], [8, 2], [9, 7], [10, 8],
        [11, 6], [12, 11], [13, 11], [14, 11], [15, 11], [16, 11], [17, 11], [18, 11],
        [19, 5], [20, 19], [21, 19], [22, 19], [23, 19], [24, 19], [25, 19], [26, 19],
    ]
    
    # Create graph args
    graph_args = {
        'num_nodes': num_nodes,
        'center': center,
        'inward_edges': inward_edges
    }
    
    # Recreate the model (same as training)
    print("Building model architecture...")
    encoder = STGCN(
        in_channels=2,
        graph_args=graph_args,
        edge_importance_weighting=True,
        dropout=dropout,
        n_out_features=256
    )
    
    decoder = FC(
        n_features=256,
        num_class=num_classes,
        dropout_ratio=dropout,
        batch_norm=False
    )
    
    model = Network(encoder=encoder, decoder=decoder)
    
    # Build the model with dummy input
    print("Building model with dummy input...")
    dummy_input = np.random.randn(1, 128, 27, 2).astype(np.float32)
    _ = model(dummy_input, training=False)
    
    # Load the weights
    print("Loading weights...")
    model.load_weights(weights_path)
    
    print("✓ Model loaded successfully!")
    return model


def predict_single(model, data, class_names):
    """
    Predict on a single sample.
    
    Args:
        model: Trained Network model
        data: (T, V, C) numpy array - single video sample
        class_names: List of class names
    
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    # Ensure data is (128, 27, 2)
    if data.shape != (128, 27, 2):
        raise ValueError(f"Expected shape (128, 27, 2), got {data.shape}")
    
    # Add batch dimension: (128, 27, 2) -> (1, 128, 27, 2)
    data_batch = np.expand_dims(data, axis=0).astype(np.float32)
    print(f"Data batch shape: {data_batch.shape}")
    
    # Get logits
    logits = model(data_batch, training=False)
    
    # Apply softmax to get probabilities
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # Get predicted class
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx]
    
    # Get top-3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_predictions = [
        {
            'class': class_names[idx],
            'confidence': float(probs[idx])
        }
        for idx in top3_indices
    ]
    
    return {
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'top3_predictions': top3_predictions,
        'all_probabilities': {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }
    }


def predict_batch(model, data_list, class_names, batch_size=32):
    """
    Predict on multiple samples.
    
    Args:
        model: Trained Network model
        data_list: List of (T, V, C) numpy arrays
        class_names: List of class names
        batch_size: Batch size for inference
    
    Returns:
        list: List of prediction results
    """
    results = []
    
    for i in range(0, len(data_list), batch_size):
        batch_data = data_list[i:i+batch_size]
        batch_array = np.stack(batch_data).astype(np.float32)
        
        # Get logits
        logits = model(batch_array, training=False)
        
        # Apply softmax
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        
        # Process each sample in batch
        for j, prob in enumerate(probs):
            pred_idx = np.argmax(prob)
            results.append({
                'predicted_class': class_names[pred_idx],
                'confidence': float(prob[pred_idx]),
                'probabilities': prob
            })
    
    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='ST-GCN Prediction for ASL Signs')
    parser.add_argument('--weights', type=str, 
                       default='../models_stgcn/stgcn_20251112_143849/best_model_weights.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--config', type=str,
                       default='../models_stgcn/stgcn_20251112_143849/config.json',
                       help='Path to config file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to preprocessed .npy file (128, 27, 2)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Show top-K predictions')
    
    args = parser.parse_args()
    
    # Load config
    print("="*70)
    print("ST-GCN PREDICTION")
    print("="*70)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    class_names = config['class_names']
    num_classes = config['num_classes']
    dropout = config['dropout']
    
    print(f"\nModel Configuration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"  Dropout: {dropout}")
    
    # Load model
    print("\n" + "="*70)
    model = load_trained_model(args.weights, num_classes, dropout)
    
    # Load data
    print("\n" + "="*70)
    print("Loading input data...")
    data = np.load(args.data).astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    if data.shape != (128, 27, 2):
        print(f"ERROR: Expected shape (128, 27, 2), got {data.shape}")
        sys.exit(1)
    
    # Make prediction
    print("\n" + "="*70)
    print("Making prediction...")
    result = predict_single(model, data, class_names)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\n✓ Predicted Sign: {result['predicted_class']}")
    print(f"✓ Confidence: {result['confidence']*100:.2f}%")
    
    print(f"\nTop-{args.top_k} Predictions:")
    for i, pred in enumerate(result['top3_predictions'][:args.top_k], 1):
        print(f"  {i}. {pred['class']:20s} - {pred['confidence']*100:6.2f}%")
    
    print("\n" + "="*70)
    print("All Class Probabilities:")
    print("="*70)
    for class_name, prob in sorted(result['all_probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{class_name:20s} {bar} {prob*100:6.2f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
