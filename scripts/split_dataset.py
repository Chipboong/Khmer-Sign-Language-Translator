"""
Re-split ASL Citizen dataset to correct train/val/test ratios.

This script reorganizes an already-split dataset to:
- Train: 60%
- Validation: 20%
- Test: 20%

Usage:
    python scripts/split_dataset.py

Directory structure expected (already split):
    dataset/
    ├── train/
    │   ├── BITE/
    │   │   ├── video1.mp4
    │   │   └── ...
    │   ├── DOG/
    │   └── ...
    ├── val/
    │   ├── BITE/
    │   └── ...
    └── test/
        ├── BITE/
        └── ...

The script will:
1. Collect ALL videos from train/val/test
2. Shuffle them
3. Re-split with correct 60/20/20 ratios
4. Clean up class names (remove "1" suffix, use "TRIP" instead of "HURDLE/TRIP1")
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Mapping from old class names to cleaned names
CLASS_NAME_MAPPING = {
	'DOG1': 'DOG',
	'HURDLE/TRIP1': 'TRIP',
	'BREAKFAST1': 'BREAKFAST',
	'DARK1': 'DARK',
	'DEMAND1': 'DEMAND',
	'BITE1': 'BITE',
	'WHATFOR1': 'WHATFOR',
	'DECIDE1': 'DECIDE',
	'ROCKINGCHAIR1': 'ROCKINGCHAIR',
	'DEAF1': 'DEAF'
}

def clean_class_name(class_name):
	"""Clean class name by removing suffix and normalizing."""
	cleaned = CLASS_NAME_MAPPING.get(class_name, class_name)
	# Ensure we return a string, not None
	return cleaned if cleaned is not None else class_name

def resplit_dataset(dataset_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Re-split an already-split dataset into correct train/val/test ratios.
    
    Args:
        dataset_dir: Path to dataset directory containing train/val/test folders
        train_ratio: Ratio for training set (default: 0.6)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.2)
        seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    dataset_path = Path(dataset_dir)
    
    # Check if train/val/test directories exist
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    test_dir = dataset_path / 'test'
    
    if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
        print("Error: Dataset must have train/, val/, and test/ directories!")
        print(f"Looking in: {dataset_path.absolute()}")
        return
    
    print("="*70)
    print("DATASET RE-SPLITTING")
    print("="*70)
    print(f"Source: {dataset_path.absolute()}")
    print(f"\nTarget ratios:")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val:   {val_ratio*100:.0f}%")
    print(f"  Test:  {test_ratio*100:.0f}%")
    print("="*70)
    
    # Get all classes from train directory (assuming all splits have same classes)
    class_names = sorted([
        d.name for d in train_dir.iterdir() 
        if d.is_dir()
    ])
    
    if not class_names:
        print("Error: No class folders found in train/ directory!")
        return
    
    print(f"\nFound {len(class_names)} classes: {class_names}")
    print("\n" + "="*70)
    print("CURRENT DISTRIBUTION")
    print("="*70)
    
    # Show current distribution
    current_stats = {'train': 0, 'val': 0, 'test': 0}
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        for class_name in class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                video_count = len(list(class_dir.glob('*.mp4')) + 
                                list(class_dir.glob('*.avi')) + 
                                list(class_dir.glob('*.mov')))
                current_stats[split_name] += video_count
    
    total_current = sum(current_stats.values())
    print(f"Total videos: {total_current}")
    print(f"  Train: {current_stats['train']} ({current_stats['train']/total_current*100:.1f}%)")
    print(f"  Val:   {current_stats['val']} ({current_stats['val']/total_current*100:.1f}%)")
    print(f"  Test:  {current_stats['test']} ({current_stats['test']/total_current*100:.1f}%)")
    
    # Create temporary directory to collect all videos
    temp_dir = dataset_path / '_temp_resplit'
    temp_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 1: Collecting all videos...")
    print("="*70)
    
    # Collect all videos from all splits, organized by class
    collected_videos = defaultdict(list)
    
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        for class_name in class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
                videos = [
                    f for f in class_dir.iterdir() 
                    if f.is_file() and f.suffix in video_extensions
                ]
                
                # Move videos to temp directory with cleaned class name
                cleaned_class_name = clean_class_name(class_name)
                temp_class_dir = temp_dir / cleaned_class_name
                temp_class_dir.mkdir(exist_ok=True)
                
                for video in videos:
                    dest = temp_class_dir / video.name
                    shutil.move(str(video), str(dest))
                    collected_videos[cleaned_class_name].append(dest)
    
    # Remove old split directories
    print("\n" + "="*70)
    print("STEP 2: Removing old split directories...")
    print("="*70)
    
    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
            print(f"✓ Removed {split_dir.name}/")
    
    # Recreate split directories
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 3: Re-splitting with new ratios...")
    print("="*70)
    
    # Statistics
    new_stats = {'train': 0, 'val': 0, 'test': 0}
    
    # Process each class
    for class_name in class_names:
        videos = collected_videos[class_name]
        
        if not videos:
            print(f"\nWarning: No videos found for {class_name}, skipping...")
            continue
        
        # Shuffle videos
        random.shuffle(videos)
        
        # Calculate split sizes
        total_videos = len(videos)
        train_size = int(total_videos * train_ratio)
        val_size = int(total_videos * val_ratio)
        # Test gets the remainder to ensure all videos are used
        test_size = total_videos - train_size - val_size
        
        # Split videos
        train_videos = videos[:train_size]
        val_videos = videos[train_size:train_size + val_size]
        test_videos = videos[train_size + val_size:]
        
        # Create class directories in each split
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Move videos to appropriate splits
        for video in train_videos:
            dest = train_dir / class_name / video.name
            shutil.move(str(video), str(dest))
            new_stats['train'] += 1
        
        for video in val_videos:
            dest = val_dir / class_name / video.name
            shutil.move(str(video), str(dest))
            new_stats['val'] += 1
        
        for video in test_videos:
            dest = test_dir / class_name / video.name
            shutil.move(str(video), str(dest))
            new_stats['test'] += 1
        
        # Print class statistics
        print(f"\n{class_name}:")
        print(f"  Total: {total_videos} videos")
        print(f"  Train: {len(train_videos)} ({len(train_videos)/total_videos*100:.1f}%)")
        print(f"  Val:   {len(val_videos)} ({len(val_videos)/total_videos*100:.1f}%)")
        print(f"  Test:  {len(test_videos)} ({len(test_videos)/total_videos*100:.1f}%)")
    
    # Remove temporary directory
    shutil.rmtree(temp_dir)
    print(f"\n✓ Cleaned up temporary directory")
    
    # Print new overall statistics
    print("\n" + "="*70)
    print("NEW DISTRIBUTION")
    print("="*70)
    total = sum(new_stats.values())
    print(f"Total videos: {total}")
    print(f"  Train: {new_stats['train']} ({new_stats['train']/total*100:.1f}%)")
    print(f"  Val:   {new_stats['val']} ({new_stats['val']/total*100:.1f}%)")
    print(f"  Test:  {new_stats['test']} ({new_stats['test']/total*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✓ DATASET RE-SPLITTING COMPLETE!")
    print("="*70)
    print("\n⚠️  NEXT STEPS:")
    print("1. Verify the new splits are correct")
    print("2. Delete old preprocessed .npy files:")
    print("   rm -rf preprocessed_npy/*")
    print("3. Run preprocess_dataset.ipynb to generate new .npy files")


def split_dataset(source_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split dataset into train/val/test sets.
    
    Args:
        source_dir: Path to dataset directory containing class folders
        train_ratio: Ratio for training set (default: 0.6)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.2)
        seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    source_path = Path(source_dir)
    
    # Get all class folders (excluding train/val/test if they already exist)
    class_folders = [
        d for d in source_path.iterdir() 
        if d.is_dir() and d.name not in ['train', 'val', 'test']
    ]
    
    if not class_folders:
        print("Error: No class folders found in source directory!")
        print(f"Looking in: {source_path.absolute()}")
        return
    
    print("="*70)
    print("DATASET SPLITTING")
    print("="*70)
    print(f"Source: {source_path.absolute()}")
    print(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
    print(f"\nSplit ratios:")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val:   {val_ratio*100:.0f}%")
    print(f"  Test:  {test_ratio*100:.0f}%")
    print("="*70)
    
    # Create output directories
    splits = {
        'train': source_path / 'train',
        'val': source_path / 'val',
        'test': source_path / 'test'
    }
    
    for split_dir in splits.values():
        split_dir.mkdir(exist_ok=True)
    
    # Statistics
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    
    # Process each class
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Get all video files in this class
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        video_files = [
            f for f in class_folder.iterdir() 
            if f.is_file() and f.suffix in video_extensions
        ]
        
        if not video_files:
            print(f"\nWarning: No videos found in {class_name}, skipping...")
            continue
        
        # Shuffle videos
        random.shuffle(video_files)
        
        # Calculate split sizes
        total_videos = len(video_files)
        train_size = int(total_videos * train_ratio)
        val_size = int(total_videos * val_ratio)
        # Test gets the remainder to ensure all videos are used
        test_size = total_videos - train_size - val_size
        
        # Split videos
        train_videos = video_files[:train_size]
        val_videos = video_files[train_size:train_size + val_size]
        test_videos = video_files[train_size + val_size:]
        
        # Create class directories in each split
        for split_name in ['train', 'val', 'test']:
            (splits[split_name] / class_name).mkdir(exist_ok=True)
        
        # Copy videos to appropriate splits
        for video in train_videos:
            dest = splits['train'] / class_name / video.name
            shutil.copy2(video, dest)
            stats[class_name]['train'] += 1
            total_stats['train'] += 1
        
        for video in val_videos:
            dest = splits['val'] / class_name / video.name
            shutil.copy2(video, dest)
            stats[class_name]['val'] += 1
            total_stats['val'] += 1
        
        for video in test_videos:
            dest = splits['test'] / class_name / video.name
            shutil.copy2(video, dest)
            stats[class_name]['test'] += 1
            total_stats['test'] += 1
        
        # Print class statistics
        print(f"\n{class_name}:")
        print(f"  Total: {total_videos} videos")
        print(f"  Train: {len(train_videos)} ({len(train_videos)/total_videos*100:.1f}%)")
        print(f"  Val:   {len(val_videos)} ({len(val_videos)/total_videos*100:.1f}%)")
        print(f"  Test:  {len(test_videos)} ({len(test_videos)/total_videos*100:.1f}%)")
    
    # Print overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    total = sum(total_stats.values())
    print(f"Total videos: {total}")
    print(f"  Train: {total_stats['train']} ({total_stats['train']/total*100:.1f}%)")
    print(f"  Val:   {total_stats['val']} ({total_stats['val']/total*100:.1f}%)")
    print(f"  Test:  {total_stats['test']} ({total_stats['test']/total*100:.1f}%)")
    
    # Print output paths
    print("\n" + "="*70)
    print("OUTPUT DIRECTORIES")
    print("="*70)
    for split_name, split_path in splits.items():
        print(f"{split_name}: {split_path.absolute()}")
    
    print("\n" + "="*70)
    print("✓ DATASET SPLITTING COMPLETE!")
    print("="*70)
    
    # Cleanup suggestion
    print("\n⚠️  NEXT STEPS:")
    print("1. Verify the splits are correct")
    print("2. Optionally delete the original class folders to save space:")
    print("   (Keep only train/val/test directories)")
    for class_folder in class_folders:
        print(f"   - {class_folder.name}/")
    print("\n   Use this command to delete them:")
    for class_folder in class_folders:
        print(f"   rm -rf dataset/{class_folder.name}")
    print("\n3. Run preprocess_dataset.ipynb to generate .npy files")


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "../dataset"  # Directory containing train/val/test folders (relative to script location)
    TRAIN_RATIO = 0.6  # 60% for training
    VAL_RATIO = 0.2    # 20% for validation
    TEST_RATIO = 0.2   # 20% for testing
    SEED = 42          # Random seed for reproducibility
    
    # Re-split the already-split dataset with correct ratios
    resplit_dataset(
        dataset_dir=DATASET_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )
