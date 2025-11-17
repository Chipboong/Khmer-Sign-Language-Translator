"""
Prepare ASL Citizen dataset for ST-GCN training.

This script:
1. Extracts keypoints from videos using MediaPipe
2. Selects 27 keypoints (11 pose + 8 right hand + 8 left hand)
3. Pads/truncates sequences to 128 frames
4. Saves in ST-GCN format: (num_frames, num_nodes, num_channels)

Output format:
    preprocessed_stgcn/
    ├── train/
    │   ├── stgcn_0000.npy  # Shape: (128, 27, 2)
    │   └── ...
    ├── val/
    │   └── ...
    ├── test/
    │   └── ...
    ├── train_labels.npy
    ├── val_labels.npy
    ├── test_labels.npy
    └── metadata.json

Usage:
    python src/prepare_stgcn_data.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import mediapipe as mp


class MediaPipeKeypointExtractor:
    """Extract keypoints from videos using MediaPipe Holistic."""
    
    def __init__(self):
        """Initialize MediaPipe Holistic."""
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints_from_video(self, video_path, target_frames=128):
        """
        Extract keypoints from video.
        
        Args:
            video_path: Path to video file
            target_frames: Target number of frames (default: 128)
        
        Returns:
            keypoints: numpy array of shape (target_frames, 543, 3)
                      543 = 33 pose + 21 left_hand + 21 right_hand + 468 face
                      3 = (x, y, z)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Sample frame indices uniformly
        if total_frames >= target_frames:
            frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            # If video is shorter, repeat frames
            frame_indices = np.arange(total_frames)
            # Pad by repeating the last frame
            frame_indices = np.pad(frame_indices, (0, target_frames - total_frames), 
                                   mode='edge')
        
        all_keypoints = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Use zeros if frame read fails
                keypoints = np.zeros((543, 2))
                all_keypoints.append(keypoints)
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.holistic.process(frame_rgb)
            
            # Extract keypoints
            keypoints = self._extract_keypoints_from_results(results)
            all_keypoints.append(keypoints)
        
        cap.release()
        
        return np.array(all_keypoints)  # (target_frames, 543, 2)
    
    def _extract_keypoints_from_results(self, results):
        """
        Extract keypoints from MediaPipe results - only x,y coordinates.
        
        Returns:
            keypoints: (543, 2) array - only x, y (no z)
                      Order: pose (33) + right_hand (21) + left_hand (21) + face (468)
        """
        keypoints = []
        
        # Pose landmarks (33)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x, lm.y])
        else:
            keypoints.extend([[0, 0]] * 33)
        
        # Right hand landmarks (21) - comes BEFORE left hand in reference code
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y])
        else:
            keypoints.extend([[0, 0]] * 21)
        
        # Left hand landmarks (21)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y])
        else:
            keypoints.extend([[0, 0]] * 21)
        
        # Face landmarks (468) - NOT USED for ST-GCN, but extracted for completeness
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                keypoints.append([lm.x, lm.y])
        else:
            keypoints.extend([[0, 0]] * 468)
        
        return np.array(keypoints)  # (543, 2)
    
    def close(self):
        """Close MediaPipe resources."""
        self.holistic.close()


def select_27_keypoints(keypoints_543):
    """
    Select 27 keypoints from 543 MediaPipe landmarks for ST-GCN.
    
    This follows the exact selection used in the reference code:
    Indices: [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
              58, 59, 62, 63, 66, 67, 70, 71, 74]
    
    From pose (0-32), left hand (33-53), and right hand (54-74).
    Note: Face landmarks (75-542) are excluded.
    
    Args:
        keypoints_543: (num_frames, 543, 2) - only x,y coordinates
    
    Returns:
        keypoints_27: (num_frames, 27, 2)
    """
    # Extract only first 75 keypoints (pose + hands, exclude face)
    # Pose: 0-32, Right hand: 33-53, Left hand: 54-74
    keypoints_75 = keypoints_543[:, :75, :]  # (num_frames, 75, 2)
    
    # Split into components
    # Note: In the extraction, order is: pose (33), right_hand (21), left_hand (21)
    # So indices are: pose 0-32, right_hand 33-53, left_hand 54-74
    posedata = keypoints_75[:, :33, :]      # (num_frames, 33, 2)
    rhdata = keypoints_75[:, 33:54, :]      # (num_frames, 21, 2)
    lhdata = keypoints_75[:, 54:75, :]      # (num_frames, 21, 2)
    
    # Concatenate in order: pose, left_hand, right_hand (to match reference)
    data = np.concatenate([posedata, lhdata, rhdata], axis=1)  # (num_frames, 75, 2)
    
    # Select the 27 keypoints using the reference indices
    # These indices now refer to the concatenated array [pose(0-32), lhand(33-53), rhand(54-74)]
    keypoints_indices = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
                        58, 59, 62, 63, 66, 67, 70, 71, 74]
    
    keypoints_27 = data[:, keypoints_indices, :]  # (num_frames, 27, 2)
    
    return keypoints_27


def normalize_keypoints(keypoints):
    """
    Normalize keypoints for ST-GCN training using shoulder-based normalization.
    
    This matches the reference normalization approach:
    1. Center on the midpoint between shoulders
    2. Scale by the mean distance between shoulders
    
    Args:
        keypoints: (num_frames, 27, 2) - after selection
    
    Returns:
        normalized: (num_frames, 27, 2) - normalized coordinates
    """
    # From the selected 27 keypoints, shoulders are at specific indices
    # Based on the selection [0, 2, 5, 11, 12, 13, 14, ...]
    # Index 3 corresponds to original pose index 11 (left_shoulder)
    # Index 4 corresponds to original pose index 12 (right_shoulder)
    shoulder_l_idx = 3  # left_shoulder in the 27-keypoint array
    shoulder_r_idx = 4  # right_shoulder in the 27-keypoint array
    
    shoulder_l = keypoints[:, shoulder_l_idx, :]  # (num_frames, 2)
    shoulder_r = keypoints[:, shoulder_r_idx, :]  # (num_frames, 2)
    
    # Calculate center as average of shoulder midpoints across all frames
    center = np.zeros(2)
    for i in range(len(shoulder_l)):
        center_i = (shoulder_r[i] + shoulder_l[i]) / 2
        center = center + center_i
    center = center / shoulder_l.shape[0]
    
    # Calculate mean distance between shoulders
    mean_dist = np.mean(np.sqrt(((shoulder_l - shoulder_r) ** 2).sum(-1)))
    
    # Normalize: center and scale
    if mean_dist != 0:
        scale = 1.0 / mean_dist
        keypoints = keypoints - center  # Center on shoulder midpoint
        keypoints = keypoints * scale    # Scale by shoulder distance
    
    return keypoints.astype(np.float32)


def load_video_paths_and_labels(data_dir, split='train'):
    """
    Load video paths and labels from dataset directory.
    
    Args:
        data_dir: Dataset directory
        split: 'train' or 'val'
    
    Returns:
        video_paths: List of video paths
        labels: List of labels (integers)
        class_names: List of class names
        class_to_idx: Dict mapping class name to index
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Get all class folders (sorted alphabetically)
    class_folders = sorted([
        d for d in os.listdir(split_dir) 
        if os.path.isdir(os.path.join(split_dir, d))
    ])
    
    print(f"\n{split.upper()} - Found {len(class_folders)} classes: {', '.join(class_folders)}")
    
    # Create mapping: class name → integer
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}
    
    # Collect videos and labels
    video_paths = []
    labels = []
    
    for class_name in class_folders:
        class_dir = os.path.join(split_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all video files
        videos = list(Path(class_dir).glob('*.mp4')) + \
                list(Path(class_dir).glob('*.avi')) + \
                list(Path(class_dir).glob('*.mov'))
        
        for video_path in videos:
            video_paths.append(str(video_path))
            labels.append(class_idx)
    
    print(f"Total videos: {len(video_paths)}")
    
    return video_paths, labels, class_folders, class_to_idx


def prepare_stgcn_dataset(data_dir, output_dir, target_frames=128):
    """
    Prepare dataset for ST-GCN training.
    
    Args:
        data_dir: Raw video dataset directory
        output_dir: Output directory for processed data
        target_frames: Target number of frames per video
    """
    print("="*70)
    print("PREPARING ST-GCN DATASET")
    print("="*70)
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Target frames: {target_frames}")
    print(f"Target nodes: 27")
    print(f"Target channels: 2 (x, y)")
    print("="*70)
    
    # Initialize MediaPipe extractor
    extractor = MediaPipeKeypointExtractor()
    
    metadata = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()}...")
        print(f"{'='*70}")
        
        # Check if split exists
        split_dir_path = os.path.join(data_dir, split)
        if not os.path.exists(split_dir_path):
            print(f"⚠️  Warning: {split} directory not found, skipping...")
            continue
        
        try:
            # Load video paths and labels
            video_paths, labels, class_names, class_to_idx = load_video_paths_and_labels(
                data_dir, split
            )
        except Exception as e:
            print(f"⚠️  Warning: Error loading {split} data: {e}")
            continue
        
        # Save metadata (from train split)
        if split == 'train':
            metadata['class_names'] = class_names
            metadata['class_to_idx'] = class_to_idx
            metadata['idx_to_class'] = {idx: name for name, idx in class_to_idx.items()}
            metadata['num_classes'] = len(class_names)
            metadata['num_frames'] = target_frames
            metadata['num_nodes'] = 27
            metadata['num_channels'] = 2
            metadata['format'] = '(num_frames, num_nodes, num_channels)'
        
        # Create output directory
        split_output_dir = Path(output_dir) / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos
        processed_labels = []
        failed_videos = []
        
        print(f"\nExtracting and processing {len(video_paths)} videos...")
        for i, (video_path, label) in enumerate(tqdm(zip(video_paths, labels), 
                                                      total=len(video_paths),
                                                      desc=f"  {split}")):
            try:
                # Extract keypoints (543 landmarks, target_frames)
                keypoints_543 = extractor.extract_keypoints_from_video(
                    video_path, 
                    target_frames=target_frames
                )  # (target_frames, 543, 2)
                
                # Select 27 keypoints (this also reorders to pose, lhand, rhand)
                keypoints_27 = select_27_keypoints(keypoints_543)  # (target_frames, 27, 2)
                
                # Normalize using shoulder-based normalization
                keypoints_normalized = normalize_keypoints(keypoints_27)  # (target_frames, 27, 2)
                
                # Verify shape
                assert keypoints_normalized.shape == (target_frames, 27, 2), \
                    f"Expected shape ({target_frames}, 27, 2), got {keypoints_normalized.shape}"
                
                # Save as NPY
                npy_path = split_output_dir / f"stgcn_{i:04d}.npy"
                np.save(npy_path, keypoints_normalized)
                
                processed_labels.append(label)
                
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                failed_videos.append(video_path)
                continue
        
        # Save labels
        labels_path = Path(output_dir) / f"{split}_labels.npy"
        np.save(labels_path, np.array(processed_labels))
        
        print(f"\n✓ Processed {len(processed_labels)}/{len(video_paths)} videos")
        print(f"✓ Keypoints saved to {split_output_dir}")
        print(f"✓ Labels saved to {labels_path}")
        
        if failed_videos:
            print(f"⚠️  Failed to process {len(failed_videos)} videos:")
            for vid in failed_videos[:5]:  # Show first 5
                print(f"  - {vid}")
            if len(failed_videos) > 5:
                print(f"  ... and {len(failed_videos) - 5} more")
    
    # Save metadata
    metadata_path = Path(output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ ST-GCN DATA PREPARATION COMPLETE!")
    print(f"✓ All data saved to: {output_dir}")
    print(f"✓ Metadata: {metadata_path}")
    print("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Classes: {metadata['num_classes']}")
    print(f"Class names: {', '.join(metadata['class_names'])}")
    print(f"\nData format:")
    print(f"  Shape per sample: ({metadata['num_frames']}, {metadata['num_nodes']}, {metadata['num_channels']})")
    print(f"  Frames: {metadata['num_frames']}")
    print(f"  Nodes: {metadata['num_nodes']} (11 pose + 8 right hand + 8 left hand)")
    print(f"  Channels: {metadata['num_channels']} (x, y)")
    print("="*70)
    
    # Close extractor
    extractor.close()
    
    return metadata


def verify_dataset(output_dir):
    """
    Verify the prepared dataset.
    
    Args:
        output_dir: Directory with prepared data
    """
    print("\n" + "="*70)
    print("VERIFYING DATASET")
    print("="*70)
    
    # Load metadata
    metadata_path = Path(output_dir) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    for split in ['train', 'val', 'test']:
        split_dir = Path(output_dir) / split
        if not split_dir.exists():
            print(f"\n{split.upper()}: Not found")
            continue
        
        # Load labels
        labels_path = Path(output_dir) / f"{split}_labels.npy"
        labels = np.load(labels_path)
        
        # Get NPY files
        npy_files = sorted(list(split_dir.glob('stgcn_*.npy')))
        
        print(f"\n{split.upper()}:")
        print(f"  Files: {len(npy_files)}")
        print(f"  Labels: {len(labels)}")
        
        if len(npy_files) > 0:
            # Load first file
            sample = np.load(npy_files[0])
            print(f"  Sample shape: {sample.shape}")
            print(f"  Sample dtype: {sample.dtype}")
            print(f"  Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
            
            # Verify shape
            expected_shape = (metadata['num_frames'], metadata['num_nodes'], metadata['num_channels'])
            if sample.shape == expected_shape:
                print(f"  ✓ Shape matches expected {expected_shape}")
            else:
                print(f"  ❌ Shape mismatch! Expected {expected_shape}, got {sample.shape}")
        
        # Label distribution
        print(f"\n  Label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for idx, count in zip(unique, counts):
            class_name = metadata['idx_to_class'][str(idx)]
            print(f"    {class_name}: {count} samples")
    
    print("\n" + "="*70)
    print("✓ VERIFICATION COMPLETE!")
    print("="*70)


def main():
    """Main function."""
    # Configuration
    DATA_DIR = "../dataset"  # Raw video dataset
    OUTPUT_DIR = "../preprocessed_stgcn"  # Output directory
    TARGET_FRAMES = 128  # Frames per video (ST-GCN standard)
    
    print("\n" + "="*70)
    print("ASL CITIZEN - ST-GCN DATA PREPARATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Input directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Target frames: {TARGET_FRAMES}")
    print(f"  Target nodes: 27 (11 pose + 8 right hand + 8 left hand)")
    print(f"  Target channels: 2 (x, y)")
    print("="*70)
    
    # Prepare dataset
    metadata = prepare_stgcn_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        target_frames=TARGET_FRAMES
    )
    
    # Verify dataset
    verify_dataset(OUTPUT_DIR)
    
    print("\n✓ Ready for ST-GCN training!")
    print(f"  Data location: {OUTPUT_DIR}")
    print(f"  Next step: Train ST-GCN model with this data")


if __name__ == "__main__":
    main()
