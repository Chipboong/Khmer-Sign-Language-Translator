"""
Predict ASL sign from a video file using trained ST-GCN model.

This script:
1. Extracts keypoints from a video file using MediaPipe
2. Processes keypoints (select 27 keypoints, normalize)
3. Loads trained ST-GCN model
4. Makes prediction and displays results

Usage:
    python predict_video.py --video path/to/video.mp4 --weights path/to/weights.h5 --config path/to/config.json
"""

import os
import sys
import json
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from existing modules
from prepare_stgcn_data import (
    MediaPipeKeypointExtractor,
    select_27_keypoints,
    normalize_keypoints
)
from predict_stgcn import load_trained_model, predict_single


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict ASL sign from video file')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--weights', type=str,
                       default='models_stgcn/stgcn_20251112_143849/best_model_weights.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--config', type=str,
                       default='models_stgcn/stgcn_20251112_143849/config.json',
                       help='Path to config file')
    parser.add_argument('--target_frames', type=int, default=128,
                       help='Target number of frames to extract')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Show top-K predictions')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("="*70)
    print("VIDEO PREDICTION - ST-GCN ASL SIGN RECOGNITION")
    print("="*70)
    print(f"\nVideo file: {args.video}")
    
    # Load config
    print(f"\nLoading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    class_names = config['class_names']
    num_classes = config['num_classes']
    dropout = config.get('dropout', 0.05)
    
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    
    # Extract keypoints from video
    print("\n" + "="*70)
    print("STEP 1: EXTRACT KEYPOINTS")
    print("="*70)
    
    extractor = MediaPipeKeypointExtractor()
    
    try:
        # Extract keypoints (543 landmarks, target_frames)
        keypoints_543 = extractor.extract_keypoints_from_video(
            args.video, 
            target_frames=args.target_frames
        )  # (target_frames, 543, 2)
        
        print(f"  Extracted keypoints shape: {keypoints_543.shape}")
        
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        extractor.close()
        sys.exit(1)
    
    # Process keypoints
    print("\n" + "="*70)
    print("STEP 2: PROCESS KEYPOINTS")
    print("="*70)
    
    # Select 27 keypoints
    keypoints_27 = select_27_keypoints(keypoints_543)  # (target_frames, 27, 2)
    print(f"  Selected 27 keypoints: {keypoints_27.shape}")
    
    # Normalize
    keypoints_normalized = normalize_keypoints(keypoints_27)  # (target_frames, 27, 2)
    print(f"  Normalized keypoints: {keypoints_normalized.shape}")
    
    # Verify shape
    assert keypoints_normalized.shape == (args.target_frames, 27, 2), \
        f"Expected shape ({args.target_frames}, 27, 2), got {keypoints_normalized.shape}"
    print("✓ Keypoint processing complete")
    
    # Close extractor
    extractor.close()
    
    # Load model
    print("\n" + "="*70)
    print("STEP 3: LOAD MODEL")
    print("="*70)
    
    model = load_trained_model(args.weights, num_classes, dropout)
    
    # Make prediction
    print("\n" + "="*70)
    print("STEP 4: PREDICT")
    print("="*70)
    print("Making prediction...")
    
    result = predict_single(model, keypoints_normalized, class_names)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\n✓ Predicted Sign: {result['predicted_class']}")
    print(f"✓ Confidence: {result['confidence']*100:.2f}%")
    
    print(f"\nTop-{args.top_k} Predictions:")
    top_predictions = result.get('top3_predictions', result.get('top5_predictions', []))
    for i, pred in enumerate(top_predictions[:args.top_k], 1):
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
    print("✓ PREDICTION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
