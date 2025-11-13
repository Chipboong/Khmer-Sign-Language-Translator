"""
Real-time ASL sign prediction using webcam with Sign Activity Detection.

This script:
1. Captures video from webcam (640x480, 30fps)
2. Detects when signing starts and stops (activity detection)
3. Extracts MediaPipe keypoints only during active signing
4. Processes keypoints (select 27 keypoints, normalize)
5. Sends captured sequence to model when signing stops
6. Displays prediction results on screen

Sign Activity Detection:
- Detects hand presence (left or right hand)
- Starts capturing when hands appear
- Stops capturing after hands disappear for idle_frames
- Simple, reliable, and responsive

Usage:
    python realtime_predict.py --weights path/to/weights.h5 --config path/to/config.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import platform

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from existing modules
from prepare_stgcn_data import (
    MediaPipeKeypointExtractor,
    select_27_keypoints,
    normalize_keypoints
)
from predict_stgcn import load_trained_model, predict_single


def draw_text_with_background(frame, text, position, font_scale=0.7, thickness=2, 
                               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x - 5, y - text_height - 5),
                  (x + text_width + 5, y + baseline + 5),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return text_height + baseline + 10


def detect_sign_activity(keypoints_543):
    """
    Detect if signing activity is happening based on hand presence.
    
    Simple and reliable: If hands are detected, signing is active.
    
    Args:
        keypoints_543: Current frame keypoints (543, 2)
    
    Returns:
        is_active: Boolean indicating if signing is detected
        hand_confidence: Float indicating hand detection confidence (0-1)
    """
    # Hand keypoints ranges:
    # Right hand: indices 33-53 (21 keypoints)
    # Left hand: indices 54-74 (21 keypoints)
    
    right_hand_keypoints = keypoints_543[33:54]  # (21, 2)
    left_hand_keypoints = keypoints_543[54:75]   # (21, 2)
    
    # Check if hands are detected (non-zero keypoints)
    right_hand_detected = np.any(right_hand_keypoints != 0)
    left_hand_detected = np.any(left_hand_keypoints != 0)
    
    # Calculate hand confidence (percentage of non-zero keypoints)
    right_hand_valid = np.sum(np.any(right_hand_keypoints != 0, axis=1)) / 21.0
    left_hand_valid = np.sum(np.any(left_hand_keypoints != 0, axis=1)) / 21.0
    
    # Hand confidence is the maximum of both hands
    hand_confidence = max(right_hand_valid, left_hand_valid)
    
    # Active if at least one hand is detected
    is_active = right_hand_detected or left_hand_detected
    
    return is_active, hand_confidence


def resample_keypoints_to_target_frames(keypoints_list, target_frames=128):
    """
    Resample variable-length keypoint sequence to target number of frames.
    
    Args:
        keypoints_list: List of keypoints, each (27, 2)
        target_frames: Target number of frames (default: 128)
    
    Returns:
        resampled: numpy array of shape (target_frames, 27, 2)
    """
    num_frames = len(keypoints_list)
    
    if num_frames == 0:
        return np.zeros((target_frames, 27, 2), dtype=np.float32)
    
    if num_frames == target_frames:
        return np.array(keypoints_list)
    
    # Convert to numpy array
    keypoints_array = np.array(keypoints_list)  # (num_frames, 27, 2)
    
    if num_frames < target_frames:
        # Pad by repeating last frame
        padding = np.repeat(keypoints_array[-1:], target_frames - num_frames, axis=0)
        resampled = np.concatenate([keypoints_array, padding], axis=0)
    else:
        # Uniformly sample target_frames from the sequence
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        resampled = keypoints_array[indices]
    
    return resampled


def main():
    """Main real-time prediction function."""
    parser = argparse.ArgumentParser(description='Real-time ASL sign prediction with activity detection')
    parser.add_argument('--weights', type=str,
                       default='models_stgcn/stgcn_20251112_143849/best_model_weights.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--config', type=str,
                       default='models_stgcn/stgcn_20251112_143849/config.json',
                       help='Path to config file')
    parser.add_argument('--target_frames', type=int, default=128,
                       help='Target number of frames for model input (will resample)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Minimum confidence to display prediction')

    parser.add_argument('--idle_frames', type=int, default=10,
                       help='Number of idle frames before considering sign complete')
    parser.add_argument('--min_frames', type=int, default=10,
                       help='Minimum frames required for a valid sign')
    parser.add_argument('--max_frames', type=int, default=200,
                       help='Maximum frames to capture for a single sign')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    class_names = config['class_names']
    num_classes = config['num_classes']
    dropout = config.get('dropout', 0.05)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {', '.join(class_names)}")
    
    # Load model
    print("\nLoading model...")
    model = load_trained_model(args.weights, num_classes, dropout)
    print("✓ Model loaded successfully")
    
    # Initialize MediaPipe extractor and drawing utilities
    print("Initializing MediaPipe...")
    extractor = MediaPipeKeypointExtractor()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    print("✓ MediaPipe initialized")
    
    # Initialize webcam
    print(f"\nOpening camera {args.camera_id}...")
    
    # Try different camera backends for cross-platform compatibility
    backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]  # DirectShow for Windows, then any available
    cap = None
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(args.camera_id, backend)
            if cap.isOpened():
                print(f"✓ Camera opened with backend: {backend}")
                break
            cap.release()
        except Exception as e:
            print(f"Failed to open camera with backend {backend}: {e}")
            continue
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open camera with any backend")
        print("Troubleshooting:")
        print("  - Check if camera is connected and not in use by another application")
        print("  - Try different --camera_id (0, 1, 2, etc.)")
        print("  - On Linux: Check camera permissions (ls -l /dev/video*)")
        extractor.close()
        sys.exit(1)
    
    if (platform.system() == 'Linux'):
        # Prefer MJPG to avoid YUYV conversion overhead/timeouts; set resolution/FPS/small buffer
        try:
            fourcc = 0
            if hasattr(cv2, "VideoWriter_fourcc"):
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Warm-up reads to start the stream
        for _ in range(10):
            cap.read()
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened successfully")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
    
    # Sign activity detection state
    is_signing = False
    idle_counter = 0
    keypoints_buffer = []  # List to store keypoints during signing
    
    # Prediction state
    current_prediction = None
    current_confidence = 0.0
    last_prediction_result = None
    
    print("\n" + "="*70)
    print("REAL-TIME PREDICTION STARTED - HAND PRESENCE DETECTION")
    print("="*70)
    print(f"Detection: Hands present = Active signing")
    print(f"Idle frames before sign end: {args.idle_frames}")
    print(f"Min/Max frames per sign: {args.min_frames}/{args.max_frames}")
    print("Press 'q' to quit, 'r' to reset")
    print("="*70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = extractor.holistic.process(frame_rgb)
            
            # Extract keypoints (543 landmarks) using the extractor's method
            keypoints_543 = extractor._extract_keypoints_from_results(results)  # (543, 2)
            
            # Detect sign activity (based on hand presence)
            is_active, hand_confidence = detect_sign_activity(keypoints_543)
            
            # State machine for sign detection
            if is_active:
                idle_counter = 0  # Reset idle counter
                
                if not is_signing:
                    # Start of new sign
                    is_signing = True
                    keypoints_buffer = []
                    print("\n🟢 Sign detected - Starting capture...")
                
                # Process and store keypoints
                keypoints_27 = select_27_keypoints(keypoints_543.reshape(1, 543, 2))[0]  # (27, 2)
                keypoints_normalized = normalize_keypoints(keypoints_27.reshape(1, 27, 2))[0]  # (27, 2)
                
                # Add to buffer if not exceeding max frames
                if len(keypoints_buffer) < args.max_frames:
                    keypoints_buffer.append(keypoints_normalized)
                
            else:
                # No activity detected
                if is_signing:
                    idle_counter += 1
                    
                    # Still capture frames during idle period (sign might not be finished)
                    if idle_counter <= args.idle_frames and len(keypoints_buffer) < args.max_frames:
                        keypoints_27 = select_27_keypoints(keypoints_543.reshape(1, 543, 2))[0]
                        keypoints_normalized = normalize_keypoints(keypoints_27.reshape(1, 27, 2))[0]
                        keypoints_buffer.append(keypoints_normalized)
                    
                    # Check if sign is complete
                    if idle_counter >= args.idle_frames:
                        is_signing = False
                        
                        # Process captured sequence
                        if len(keypoints_buffer) >= args.min_frames:
                            print(f"🔵 Sign complete - Captured {len(keypoints_buffer)} frames, processing...")
                            
                            # Resample to target frames
                            keypoints_sequence = resample_keypoints_to_target_frames(
                                keypoints_buffer, target_frames=args.target_frames
                            )
                            
                            # Make prediction
                            try:
                                result = predict_single(model, keypoints_sequence, class_names)
                                current_prediction = result['predicted_class']
                                current_confidence = result['confidence']
                                last_prediction_result = result
                                
                                print(f"✅ Prediction: {current_prediction} ({current_confidence*100:.1f}%)")
                            except Exception as e:
                                print(f"❌ Prediction error: {e}")
                        else:
                            print(f"⚠️  Sign too short - Only {len(keypoints_buffer)} frames (min: {args.min_frames})")
                        
                        # Clear buffer
                        keypoints_buffer = []
                        idle_counter = 0
            
            # Draw MediaPipe landmarks on frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )
            
            # Draw UI elements
            y_offset = 30
            
            # Activity status
            if is_signing:
                status_text = f"SIGNING - Frames: {len(keypoints_buffer)}/{args.max_frames}"
                status_color = (0, 0, 255)  # Red
                status_bg = (0, 0, 100)
            else:
                if is_active:
                    status_text = f"HANDS DETECTED - Confidence: {hand_confidence:.2f}"
                    status_color = (0, 255, 255)  # Yellow
                    status_bg = (50, 50, 0)
                else:
                    status_text = f"IDLE - No hands detected"
                    status_color = (200, 200, 200)  # Gray
                    status_bg = (50, 50, 50)
            
            y_offset += draw_text_with_background(frame, status_text, (10, y_offset), 
                                                  font_scale=0.7, thickness=2,
                                                  text_color=status_color, 
                                                  bg_color=status_bg)
            
            # Draw prediction if confidence is high enough
            if current_prediction and current_confidence >= args.confidence_threshold:
                y_offset += 10
                
                # Prediction box
                pred_text = f"Sign: {current_prediction}"
                y_offset += draw_text_with_background(frame, pred_text, (10, y_offset),
                                                      font_scale=1.0, thickness=2,
                                                      text_color=(0, 255, 0), 
                                                      bg_color=(0, 0, 0))
                
                # Confidence
                conf_text = f"Confidence: {current_confidence*100:.1f}%"
                y_offset += draw_text_with_background(frame, conf_text, (10, y_offset),
                                                      font_scale=0.8, thickness=2,
                                                      text_color=(0, 255, 255), 
                                                      bg_color=(0, 0, 0))
            
            # Show top predictions if available
            if last_prediction_result and not is_signing:
                y_offset += 10
                top_preds_text = "Top 3 Predictions:"
                y_offset += draw_text_with_background(frame, top_preds_text, (10, y_offset),
                                                      font_scale=0.6, thickness=1,
                                                      text_color=(150, 150, 150),
                                                      bg_color=(0, 0, 0))
                
                top_predictions = last_prediction_result.get('top3_predictions', [])
                for i, pred in enumerate(top_predictions[:3], 1):
                    pred_line = f"  {i}. {pred['class']} - {pred['confidence']*100:.1f}%"
                    y_offset += draw_text_with_background(frame, pred_line, (10, y_offset),
                                                          font_scale=0.5, thickness=1,
                                                          text_color=(150, 150, 150),
                                                          bg_color=(0, 0, 0))
            
            # Instructions
            instructions = [
                "Start signing to begin capture",
                "Press 'q' to quit | 'r' to reset"
            ]
            
            inst_y = frame.shape[0] - 20 * len(instructions) - 10
            for instruction in instructions:
                inst_y += draw_text_with_background(frame, instruction, (10, inst_y),
                                                   font_scale=0.5, thickness=1,
                                                   text_color=(200, 200, 200),
                                                   bg_color=(0, 0, 0))
            
            # Display frame (with error handling for platforms without GUI support)
            try:
                cv2.imshow('ASL Real-time Prediction', frame)
            except cv2.error as e:
                print(f"\nError: Cannot display video window. OpenCV built without GUI support.")
                print("This script requires OpenCV with GUI support (highgui module).")
                print("Install with: pip install opencv-python (not opencv-python-headless)")
                break
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\n🔄 Resetting...")
                keypoints_buffer = []
                current_prediction = None
                current_confidence = 0.0
                last_prediction_result = None
                is_signing = False
                idle_counter = 0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        
        # Destroy windows with error handling for different platforms
        try:
            cv2.destroyAllWindows()
        except cv2.error as e:
            # Some OpenCV builds don't support destroyAllWindows()
            # This is common in headless or minimal installations
            print(f"Note: Could not destroy windows (this is normal for some OpenCV builds)")
        except Exception as e:
            print(f"Note: Window cleanup warning: {e}")
        
        extractor.close()
        print("✓ Cleanup complete")
        print("\n" + "="*70)
        print("REAL-TIME PREDICTION STOPPED")
        print("="*70)


if __name__ == "__main__":
    main()
