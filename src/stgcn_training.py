"""
Training script for ST-GCN on ASL Citizen dataset.

This script:
1. Loads preprocessed ST-GCN data (128 frames, 27 nodes, 2 channels)
2. Applies data augmentation during training
3. Trains ST-GCN model with proper callbacks
4. Saves best model and training history

Usage:
    python src/stgcn_training.py
"""

import os
import sys

# Suppress TensorFlow warnings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Additional logging suppression
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Import ST-GCN model (NEW: using PyTorch-equivalent architecture)
from models.stgcn_tf import STGCN, FC, Network
from models.graph_utils import GraphWithPartition

# Import augmentation
from stgcn_augmentation import get_default_augmentation


class STGCNDataGenerator(keras.utils.Sequence):
    """
    Data generator for ST-GCN training.
    
    Loads preprocessed .npy files and applies augmentation.
    
    Args:
        data_dir (str): Directory with preprocessed data
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        augmentation (bool): Whether to apply augmentation
    """
    
    def __init__(self, data_dir, split='train', batch_size=32, shuffle=True, augmentation=True):
        """Initialize data generator."""
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load labels
        labels_path = self.data_dir / f'{split}_labels.npy'
        self.labels = np.load(labels_path)
        
        # Get file paths
        split_dir = self.data_dir / split
        self.file_paths = sorted(list(split_dir.glob('stgcn_*.npy')))
        
        # Verify lengths match
        assert len(self.file_paths) == len(self.labels), \
            f"Mismatch: {len(self.file_paths)} files vs {len(self.labels)} labels"
        
        # Create augmentor
        if self.augmentation and split == 'train':
            self.augmentor = get_default_augmentation(training=True)
        else:
            self.augmentor = get_default_augmentation(training=False)
        
        # Shuffle indices
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        print(f"{split.upper()} Generator: {len(self.file_paths)} samples, {len(self)} batches")
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Get batch.
        
        Args:
            idx: Batch index
        
        Returns:
            X: Batch of data (batch, time, vertices, channels)
            y: Batch of labels (batch, num_classes) - one-hot encoded
        """
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load batch
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            # Load data: (128, 27, 2)
            data = np.load(self.file_paths[i]).astype(np.float32)
            
            # Convert to (C, T, V) format for augmentation
            # (128, 27, 2) -> (2, 128, 27)
            data_ctv = np.transpose(data, (2, 0, 1))  # (T, V, C) -> (C, T, V)
            
            # Apply augmentation
            if self.augmentation and self.split == 'train':
                data_ctv = self.augmentor(data_ctv)
                # Convert back to numpy if tensor
                if hasattr(data_ctv, 'numpy'):
                    data_ctv = data_ctv.numpy()
            
            # Convert back to (T, V, C) format for model
            # (2, 128, 27) -> (128, 27, 2)
            data_tvc = np.transpose(data_ctv, (1, 2, 0))  # (C, T, V) -> (T, V, C)
            
            X_batch.append(data_tvc)
            y_batch.append(self.labels[i])
        
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # One-hot encode labels
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.metadata['num_classes'])
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Shuffle indices at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_model(num_classes, dropout=0.05):
    """
    Create ST-GCN model (NEW: using PyTorch-equivalent architecture).
    
    This creates the Network wrapper with:
    - STGCN encoder: 10 blocks with edge importance weighting
    - FC decoder: Fully connected classifier head
    
    Args:
        num_classes (int): Number of classes
        dropout (float): Dropout rate (default 0.05 to match reference)
    
    Returns:
        model: Network model (encoder-decoder)
    """
    # Define ASL skeleton graph (27 nodes)
    # Connections based on ASL Citizen MediaPipe Holistic landmarks
    # From src/models/graph.py
    num_nodes = 27
    center = 0  # Nose as center
    
    # Define skeleton edges (inward from periphery to center)
    # Following the kinematic chain structure from graph.py
    inward_edges = [
        # Pose edges - torso and upper body
        [1, 0],   # left_shoulder -> nose
        [2, 0],   # right_shoulder -> nose  
        [3, 1],   # left_elbow -> left_shoulder
        [4, 2],   # right_elbow -> right_shoulder
        [5, 3],   # left_wrist -> left_elbow
        [6, 4],   # right_wrist -> right_elbow
        
        # Lower body
        [7, 1],   # left_hip -> left_shoulder
        [8, 2],   # right_hip -> right_shoulder
        [9, 7],   # left_knee -> left_hip
        [10, 8],  # right_knee -> right_hip
        
        # Right hand (11-18)
        [11, 6],  # hand_right_wrist -> pose_right_wrist
        [12, 11], # thumb_tip -> wrist
        [13, 11], # index_tip -> wrist
        [14, 11], # middle_tip -> wrist
        [15, 11], # ring_tip -> wrist
        [16, 11], # pinky_tip -> wrist
        [17, 11], # index_mcp -> wrist
        [18, 11], # middle_mcp -> wrist
        
        # Left hand (19-26)
        [19, 5],  # hand_left_wrist -> pose_left_wrist
        [20, 19], # thumb_tip -> wrist
        [21, 19], # index_tip -> wrist
        [22, 19], # middle_tip -> wrist
        [23, 19], # ring_tip -> wrist
        [24, 19], # pinky_tip -> wrist
        [25, 19], # index_mcp -> wrist
        [26, 19], # middle_mcp -> wrist
    ]
    
    # Create graph args (will be used by STGCN to build graph internally)
    graph_args = {
        'num_nodes': num_nodes,
        'center': center,
        'inward_edges': inward_edges
    }
    
    # Create encoder (STGCN)
    encoder = STGCN(
        in_channels=2,  # x, y coordinates
        graph_args=graph_args,
        edge_importance_weighting=True,
        dropout=dropout,
        n_out_features=256
    )
    
    # Create decoder (FC classifier)
    decoder = FC(
        n_features=256,  # Output from STGCN
        num_class=num_classes,  # Note: parameter name is num_class (singular)
        dropout_ratio=dropout,
        batch_norm=False  # Reference doesn't use BN in FC
    )
    
    # Wrap in Network
    model = Network(encoder=encoder, decoder=decoder)
    
    return model


def plot_training_history(history, save_path):
    """
    Plot and save training history.
    
    Args:
        history: Training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved to {save_path}")


def main():
    """Main training function."""
    print("="*70)
    print("ST-GCN TRAINING ON ASL CITIZEN DATASET")
    print("="*70)
    
    # Configuration (matching reference PyTorch implementation)
    DATA_DIR = "../preprocessed_stgcn"
    OUTPUT_DIR = "../models_stgcn"
    BATCH_SIZE = 32  # Changed from 16 to match reference
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    DROPOUT = 0.05  # Changed from 0.5 to match reference (critical fix!)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(OUTPUT_DIR, f"stgcn_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {model_dir}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Dropout: {DROPOUT}")
    print("="*70)
    
    # Load metadata
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    
    print(f"\nDataset Info:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"  Data format: {metadata['format']}")
    print(f"  Frames: {metadata['num_frames']}")
    print(f"  Nodes: {metadata['num_nodes']}")
    print(f"  Channels: {metadata['num_channels']}")
    
    # Create data generators
    print("\n" + "="*70)
    print("Creating Data Generators...")
    print("="*70)
    
    train_gen = STGCNDataGenerator(
        DATA_DIR,
        split='train',
        batch_size=BATCH_SIZE,
        shuffle=True,
        augmentation=True
    )
    
    val_gen = STGCNDataGenerator(
        DATA_DIR,
        split='val',
        batch_size=BATCH_SIZE,
        shuffle=False,
        augmentation=False
    )
    
    # Check if test split exists
    test_labels_path = os.path.join(DATA_DIR, 'test_labels.npy')
    if os.path.exists(test_labels_path):
        test_gen = STGCNDataGenerator(
            DATA_DIR,
            split='test',
            batch_size=BATCH_SIZE,
            shuffle=False,
            augmentation=False
        )
        print(f"Test generator created: {len(test_gen.file_paths)} samples")
    else:
        test_gen = None
        print("No test split found, skipping test evaluation")
    
    # Create model
    print("\n" + "="*70)
    print("Building ST-GCN Model...")
    print("="*70)
    
    model = create_model(num_classes=num_classes, dropout=DROPOUT)
    
    # Build model
    model.build(input_shape=(None, metadata['num_frames'], metadata['num_nodes'], metadata['num_channels']))
    
    # Learning rate schedule (Cosine Annealing like reference)
    steps_per_epoch = len(train_gen)
    total_steps = steps_per_epoch * EPOCHS
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=0.0  # Minimum learning rate
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    print("\n" + "="*70)
    print("Setting up Callbacks...")
    print("="*70)
    
    callbacks = [
        # Model checkpoint (save best model weights only)
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model_weights.weights.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,  # Changed to True to avoid serialization issues
            mode='max',
            verbose=1,
            save_freq='epoch'
        ),
        
        # Early stopping (increased patience since we have better hyperparameters)
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,  # Increased from 20
            restore_best_weights=True,
            verbose=1
        ),
        
        # Note: Learning rate scheduler now using CosineDecay (not ReduceLROnPlateau)
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            filename=os.path.join(model_dir, 'training_log.csv'),
            append=False
        )
    ]
    
    print("✓ Callbacks configured")
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    # Plot training history
    plot_path = os.path.join(model_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION EVALUATION")
    print("="*70)
    
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Evaluate on test set if available
    if test_gen is not None:
        print("\n" + "="*70)
        print("TEST EVALUATION")
        print("="*70)
        
        test_loss, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save final model weights
    final_weights_path = os.path.join(model_dir, 'final_model_weights.weights.h5')
    model.save_weights(final_weights_path)
    print(f"\n✓ Final model weights saved to {final_weights_path}")
    
    # Save training configuration
    config = {
        'num_classes': num_classes,
        'class_names': class_names,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'dropout': DROPOUT,
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
    }
    
    if test_gen is not None:
        config['test_accuracy'] = float(test_acc)
        config['test_loss'] = float(test_loss)
    
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration saved to {config_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model directory: {model_dir}")
    print(f"Best model weights: {os.path.join(model_dir, 'best_model_weights.weights.h5')}")
    print(f"Final model weights: {final_weights_path}")
    print(f"Training history plot: {plot_path}")
    print(f"Configuration: {config_path}")
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
    if test_gen is not None:
        print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print("="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
