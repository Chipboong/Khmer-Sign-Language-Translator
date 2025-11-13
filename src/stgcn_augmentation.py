"""
Data augmentation for ST-GCN skeleton sequences.

This module provides augmentation transforms for skeleton-based action recognition.
All transforms work with TensorFlow tensors.

Input format: (C, T, V) where:
    C = Channels (2 for x,y coordinates)
    T = Temporal frames (128)
    V = Vertices/nodes (27 keypoints)

Usage:
    augmentor = Compose([
        ShearTransform(shear_std=0.2),
        RotationTransform(rotation_std=0.2)
    ])
    
    augmented_data = augmentor(data)  # data shape: (2, 128, 27)
"""

import tensorflow as tf
import numpy as np


class Compose:
    """
    Compose a list of pose transforms.
    
    Args:
        transforms (list): List of transforms to be applied sequentially.
    
    Example:
        >>> transforms = Compose([
        ...     ShearTransform(shear_std=0.2),
        ...     RotationTransform(rotation_std=0.2)
        ... ])
        >>> augmented = transforms(data)
    """

    def __init__(self, transforms):
        """
        Initialize Compose with a list of transforms.
        
        Args:
            transforms (list): List of transform objects.
        """
        self.transforms = transforms

    def __call__(self, x):
        """
        Apply all transforms sequentially.

        Args:
            x (tf.Tensor): Input data with shape (C, T, V)
                          C = Channels (2)
                          T = Temporal frames (128)
                          V = Vertices/nodes (27)

        Returns:
            tf.Tensor: Augmented data with same shape (C, T, V)
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class ShearTransform:
    """
    Applies 2D shear transformation to skeleton sequences.
    
    Shear matrix: [[1, shear], [0, 1]]
    
    This transformation shifts x-coordinates proportionally to y-coordinates,
    creating a "slanting" effect on the skeleton.
    
    Args:
        shear_std (float): Standard deviation for shear transformation. 
                          Shear value is sampled from N(0, shear_std). 
                          Default: 0.2
        probability (float): Probability of applying this transform. Default: 0.5
    
    Reference:
        https://en.wikipedia.org/wiki/Shear_matrix
    """
    
    def __init__(self, shear_std: float = 0.2, probability: float = 0.5):
        """
        Initialize ShearTransform.
        
        Args:
            shear_std (float): Standard deviation for shear amount.
            probability (float): Probability of applying transform.
        """
        self.shear_std = shear_std
        self.probability = probability

    def __call__(self, data):
        """
        Apply shear transformation to the given skeleton data.

        Args:
            data (tf.Tensor or np.ndarray): Input skeleton sequence
                                            Shape: (C, T, V) where
                                            C = 2 (x, y channels)
                                            T = 128 (temporal frames)
                                            V = 27 (vertices/keypoints)

        Returns:
            tf.Tensor: Sheared skeleton data with shape (C, T, V)
        """
        # Skip augmentation with probability
        if np.random.rand() > self.probability:
            return data
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = tf.constant(data, dtype=tf.float32)
        
        # Ensure float32 for matrix operations
        x = tf.cast(data, tf.float32)
        
        # Verify input shape
        assert x.shape[0] == 2, f"Only 2 channels supported for ShearTransform, got {x.shape[0]}"
        
        # Transpose: (C, T, V) -> (T, V, C)
        # C = Channels, T = Temporal, V = Vertices
        x = tf.transpose(x, perm=[1, 2, 0])  # (C, T, V) -> (T, V, C)
        
        # Create shear matrix: [[1, shear], [0, 1]]
        shear_value = np.random.normal(loc=0.0, scale=self.shear_std, size=1)[0]
        shear_matrix = tf.constant([
            [1.0, shear_value],
            [0.0, 1.0]
        ], dtype=tf.float32)
        
        # Apply shear: (T, V, C) @ (C, C) = (T, V, C)
        # For each frame and vertex, transform the (x, y) coordinates
        res = tf.linalg.matmul(x, shear_matrix)
        
        # Transpose back: (T, V, C) -> (C, T, V)
        result = tf.transpose(res, perm=[2, 0, 1])  # (T, V, C) -> (C, T, V)
        
        return result


class RotationTransform:
    """
    Applies 2D rotation transformation to skeleton sequences.
    
    Rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    
    This transformation rotates the skeleton around the origin,
    useful for making the model invariant to viewing angle.
    
    Args:
        rotation_std (float): Standard deviation for rotation angle (in radians).
                             Angle is sampled from N(0, rotation_std).
                             Default: 0.2 (~11 degrees)
        probability (float): Probability of applying this transform. Default: 0.5
    
    Reference:
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    
    def __init__(self, rotation_std: float = 0.2, probability: float = 0.5):
        """
        Initialize RotationTransform.
        
        Args:
            rotation_std (float): Standard deviation for rotation angle in radians.
            probability (float): Probability of applying transform.
        """
        self.rotation_std = rotation_std
        self.probability = probability

    def __call__(self, data):
        """
        Apply rotation transformation to the given skeleton data.

        Args:
            data (tf.Tensor or np.ndarray): Input skeleton sequence
                                            Shape: (C, T, V) where
                                            C = 2 (x, y channels)
                                            T = 128 (temporal frames)
                                            V = 27 (vertices/keypoints)

        Returns:
            tf.Tensor: Rotated skeleton data with shape (C, T, V)
        """
        # Skip augmentation with probability
        if np.random.rand() > self.probability:
            return data
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = tf.constant(data, dtype=tf.float32)
        
        # Ensure float32 for matrix operations
        x = tf.cast(data, tf.float32)
        
        # Verify input shape
        assert x.shape[0] == 2, f"Only 2 channels supported for RotationTransform, got {x.shape[0]}"
        
        # Transpose: (C, T, V) -> (T, V, C)
        # C = Channels, T = Temporal, V = Vertices
        x = tf.transpose(x, perm=[1, 2, 0])  # (C, T, V) -> (T, V, C)
        
        # Sample rotation angle from normal distribution
        rotation_angle = np.random.normal(loc=0.0, scale=self.rotation_std, size=1)[0]
        
        # Create rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        rotation_cos = np.cos(rotation_angle)
        rotation_sin = np.sin(rotation_angle)
        rotation_matrix = tf.constant([
            [rotation_cos, -rotation_sin],
            [rotation_sin, rotation_cos]
        ], dtype=tf.float32)
        
        # Apply rotation: (T, V, C) @ (C, C) = (T, V, C)
        # For each frame and vertex, transform the (x, y) coordinates
        res = tf.linalg.matmul(x, rotation_matrix)
        
        # Transpose back: (T, V, C) -> (C, T, V)
        result = tf.transpose(res, perm=[2, 0, 1])  # (T, V, C) -> (C, T, V)
        
        return result

# Default augmentation pipeline for ST-GCN training
def get_default_augmentation(training=True):
    """
    Get default augmentation pipeline for ST-GCN.
    
    MATCHES REFERENCE PYTORCH IMPLEMENTATION:
    Only shear and rotation with 0.1 std (10% variation)
    
    Args:
        training (bool): If True, return training augmentations.
                        If False, return no augmentation (for validation/test).
    
    Returns:
        Compose: Augmentation pipeline
    
    Example:
        >>> train_aug = get_default_augmentation(training=True)
        >>> val_aug = get_default_augmentation(training=False)
        >>> 
        >>> # During training
        >>> augmented_data = train_aug(skeleton_data)
    """
    if training:
        return Compose([
            ShearTransform(shear_std=0.1, probability=1.0),      # Always apply, 10% variation
            RotationTransform(rotation_std=0.1, probability=1.0),  # Always apply, 10% variation
        ])
    else:
        # No augmentation for validation/test
        return Compose([])


# if __name__ == "__main__":
#     """Test augmentation transforms."""
#     print("="*70)
#     print("ST-GCN DATA AUGMENTATION TEST")
#     print("="*70)
    
#     # Create dummy data: (C, T, V) = (2, 128, 27)
#     dummy_data = np.random.randn(2, 128, 27).astype(np.float32)
#     print(f"\nOriginal data shape: {dummy_data.shape}")
#     print(f"Original data range: [{dummy_data.min():.3f}, {dummy_data.max():.3f}]")
    
#     # Test individual transforms
#     print("\n" + "-"*70)
#     print("Testing Individual Transforms:")
#     print("-"*70)
    
#     # Shear
#     shear = ShearTransform(shear_std=0.2, probability=1.0)
#     sheared = shear(dummy_data)
#     print(f"Shear: {sheared.shape} | range: [{sheared.numpy().min():.3f}, {sheared.numpy().max():.3f}]")
    
#     # Rotation
#     rotation = RotationTransform(rotation_std=0.2, probability=1.0)
#     rotated = rotation(dummy_data)
#     print(f"Rotation: {rotated.shape} | range: [{rotated.numpy().min():.3f}, {rotated.numpy().max():.3f}]")
    
#     # Scale
#     scale = ScaleTransform(scale_std=0.1, probability=1.0)
#     scaled = scale(dummy_data)
#     print(f"Scale: {scaled.shape} | range: [{scaled.numpy().min():.3f}, {scaled.numpy().max():.3f}]")
    
#     # Translation
#     translation = TranslationTransform(translation_std=0.05, probability=1.0)
#     translated = translation(dummy_data)
#     print(f"Translation: {translated.shape} | range: [{translated.numpy().min():.3f}, {translated.numpy().max():.3f}]")
    
#     # Temporal Crop
#     temporal_crop = TemporalCropTransform(crop_ratio_range=(0.8, 1.0), probability=1.0)
#     cropped = temporal_crop(dummy_data)
#     print(f"Temporal Crop: {cropped.shape} | range: [{cropped.numpy().min():.3f}, {cropped.numpy().max():.3f}]")
    
#     # Test full pipeline
#     print("\n" + "-"*70)
#     print("Testing Full Augmentation Pipeline:")
#     print("-"*70)
    
#     augmentor = get_default_augmentation(training=True)
#     augmented = augmentor(dummy_data)
#     print(f"Augmented data shape: {augmented.shape}")
#     print(f"Augmented data range: [{augmented.numpy().min():.3f}, {augmented.numpy().max():.3f}]")
    
#     print("\n" + "="*70)
#     print("✓ ALL TESTS PASSED!")
#     print("="*70)
