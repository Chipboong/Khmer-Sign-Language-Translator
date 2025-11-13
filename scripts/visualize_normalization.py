"""
Visualize shoulder-based normalization for skeleton keypoints.

This script demonstrates how the normalization works by showing:
1. Original MediaPipe coordinates (0-1 range)
2. After shoulder-based normalization (centered and scaled)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def create_example_skeleton():
    """
    Create example skeleton keypoints in MediaPipe format (0-1 range).
    
    Returns:
        keypoints: (27, 2) array with x, y coordinates
    """
    # Simulating a person making a sign
    # These are approximate positions for visualization
    keypoints = np.array([
        # Pose keypoints (7 selected points)
        [0.50, 0.20],  # 0: nose
        [0.45, 0.35],  # 1: left_shoulder  
        [0.55, 0.35],  # 2: right_shoulder
        [0.42, 0.50],  # 3: left_elbow
        [0.58, 0.50],  # 4: right_elbow
        [0.40, 0.65],  # 5: left_wrist
        [0.60, 0.65],  # 6: right_wrist
        
        # Left hand keypoints (10 points) - raised hand
        [0.40, 0.65],  # 7: left_wrist
        [0.38, 0.70],  # 8: left_thumb_tip
        [0.39, 0.72],  # 9: left_index_tip
        [0.40, 0.73],  # 10: left_middle_tip
        [0.41, 0.72],  # 11: left_ring_tip
        [0.42, 0.70],  # 12: left_pinky_tip
        [0.39, 0.68],  # 13: left_index_mcp
        [0.40, 0.68],  # 14: left_middle_mcp
        [0.41, 0.68],  # 15: left_ring_mcp
        [0.42, 0.67],  # 16: left_pinky_mcp
        
        # Right hand keypoints (10 points) - lowered hand
        [0.60, 0.65],  # 17: right_wrist
        [0.62, 0.68],  # 18: right_thumb_tip
        [0.61, 0.70],  # 19: right_index_tip
        [0.60, 0.71],  # 20: right_middle_tip
        [0.59, 0.70],  # 21: right_ring_tip
        [0.58, 0.68],  # 22: right_pinky_tip
        [0.61, 0.67],  # 23: right_index_mcp
        [0.60, 0.67],  # 24: right_middle_mcp
        [0.59, 0.67],  # 25: right_ring_mcp
        [0.58, 0.66],  # 26: right_pinky_mcp
    ])
    
    return keypoints


def shoulder_normalize(keypoints):
    """
    Apply shoulder-based normalization.
    
    Args:
        keypoints: (27, 2) array
    
    Returns:
        normalized: (27, 2) array
    """
    # Shoulder indices in our 27-keypoint array
    shoulder_l_idx = 1  # left_shoulder
    shoulder_r_idx = 2  # right_shoulder
    
    shoulder_l = keypoints[shoulder_l_idx]
    shoulder_r = keypoints[shoulder_r_idx]
    
    # Calculate center (midpoint between shoulders)
    center = (shoulder_l + shoulder_r) / 2
    
    # Calculate shoulder distance
    shoulder_dist = np.sqrt(((shoulder_l - shoulder_r) ** 2).sum())
    
    # Normalize
    if shoulder_dist != 0:
        scale = 1.0 / shoulder_dist
        normalized = (keypoints - center) * scale
    else:
        normalized = keypoints - center
    
    return normalized, center, shoulder_dist


def plot_skeleton(ax, keypoints, title, color='blue', show_grid=True):
    """Plot skeleton keypoints with connections."""
    
    # Define connections for visualization
    # Pose connections
    pose_connections = [
        (0, 1), (0, 2),  # nose to shoulders
        (1, 2),          # shoulder to shoulder
        (1, 3), (3, 5),  # left arm
        (2, 4), (4, 6),  # right arm
    ]
    
    # Hand connections (simplified - just wrist to fingers)
    left_hand_connections = [(7, i) for i in range(8, 17)]
    right_hand_connections = [(17, i) for i in range(18, 27)]
    
    all_connections = pose_connections + left_hand_connections + right_hand_connections
    
    # Draw connections
    for i, j in all_connections:
        xs = [keypoints[i, 0], keypoints[j, 0]]
        ys = [keypoints[i, 1], keypoints[j, 1]]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6)
    
    # Draw keypoints
    ax.scatter(keypoints[:7, 0], keypoints[:7, 1], 
               c='red', s=100, zorder=3, label='Pose', edgecolors='black', linewidths=1)
    ax.scatter(keypoints[7:17, 0], keypoints[7:17, 1], 
               c='green', s=60, zorder=3, label='Left Hand', edgecolors='black', linewidths=1)
    ax.scatter(keypoints[17:, 0], keypoints[17:, 1], 
               c='blue', s=60, zorder=3, label='Right Hand', edgecolors='black', linewidths=1)
    
    # Highlight shoulders
    ax.scatter(keypoints[1:3, 0], keypoints[1:3, 1], 
               c='yellow', s=150, zorder=4, marker='*', 
               edgecolors='black', linewidths=2, label='Shoulders')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')


def main():
    """Main visualization function."""
    print("="*70)
    print("SHOULDER-BASED NORMALIZATION VISUALIZATION")
    print("="*70)
    
    # Create example skeleton
    original_keypoints = create_example_skeleton()
    print(f"\nOriginal keypoints shape: {original_keypoints.shape}")
    print(f"Original X range: [{original_keypoints[:, 0].min():.3f}, {original_keypoints[:, 0].max():.3f}]")
    print(f"Original Y range: [{original_keypoints[:, 1].min():.3f}, {original_keypoints[:, 1].max():.3f}]")
    
    # Apply shoulder normalization
    normalized_keypoints, center, shoulder_dist = shoulder_normalize(original_keypoints)
    print(f"\nShoulder center: ({center[0]:.3f}, {center[1]:.3f})")
    print(f"Shoulder distance: {shoulder_dist:.3f}")
    print(f"Scale factor: {1.0/shoulder_dist:.3f}")
    
    print(f"\nNormalized X range: [{normalized_keypoints[:, 0].min():.3f}, {normalized_keypoints[:, 0].max():.3f}]")
    print(f"Normalized Y range: [{normalized_keypoints[:, 1].min():.3f}, {normalized_keypoints[:, 1].max():.3f}]")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Plot 1: Original coordinates (MediaPipe 0-1 range)
    ax1 = plt.subplot(1, 2, 1)
    plot_skeleton(ax1, original_keypoints, 
                  'Original MediaPipe Coordinates\n(Absolute position in frame, 0-1 range)',
                  color='steelblue')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.invert_yaxis()  # Invert Y to match image coordinates
    
    # Add frame boundary
    rect = mpatches.Rectangle((0, 0), 1, 1, linewidth=2, 
                              edgecolor='black', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.text(0.5, -0.05, 'Frame Boundary (0-1)', 
            ha='center', fontsize=10, style='italic')
    
    # Plot 2: Normalized coordinates (shoulder-based)
    ax2 = plt.subplot(1, 2, 2)
    plot_skeleton(ax2, normalized_keypoints, 
                  'Shoulder-Based Normalized Coordinates\n(Relative to body center, scaled by shoulder width)',
                  color='forestgreen')
    
    # Set symmetric limits around origin
    max_range = max(abs(normalized_keypoints).max(), 2.0)
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.invert_yaxis()  # Invert Y to match image coordinates
    
    # Draw origin (shoulder center)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Center (0,0)')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.plot(0, 0, 'r*', markersize=20, label='Origin (Shoulder Center)', zorder=5)
    
    # Draw reference circle (shoulder distance = 1.0)
    circle = plt.Circle((0, 0), 1.0, color='orange', fill=False, 
                       linestyle=':', linewidth=2, label='Shoulder distance = 1.0')
    ax2.add_patch(circle)
    
    # Add annotations
    ax2.text(0, -max_range * 0.9, 
            'Note: Negative values are normal!\nCoordinates are relative to body center.',
            ha='center', fontsize=10, style='italic', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = '../visualization_normalization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print detailed comparison
    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)
    print(f"{'Keypoint':<20} {'Original (x, y)':<20} {'Normalized (x, y)':<20}")
    print("-"*70)
    
    keypoint_names = [
        'Nose', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
        'LH_Wrist', 'LH_Thumb', 'LH_Index', 'LH_Middle', 'LH_Ring', 'LH_Pinky',
        'LH_Index_MCP', 'LH_Middle_MCP', 'LH_Ring_MCP', 'LH_Pinky_MCP',
        'RH_Wrist', 'RH_Thumb', 'RH_Index', 'RH_Middle', 'RH_Ring', 'RH_Pinky',
        'RH_Index_MCP', 'RH_Middle_MCP', 'RH_Ring_MCP', 'RH_Pinky_MCP'
    ]
    
    for i, name in enumerate(keypoint_names[:10]):  # Show first 10 for brevity
        orig = original_keypoints[i]
        norm = normalized_keypoints[i]
        print(f"{name:<20} ({orig[0]:.3f}, {orig[1]:.3f}){'':<8} "
              f"({norm[0]:+.3f}, {norm[1]:+.3f})")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Shoulders are now at (-0.5, 0.0) and (+0.5, 0.0)")
    print("   → Distance between shoulders = 1.0 (our reference unit)")
    print("\n2. Center (0, 0) is at the midpoint between shoulders")
    print("   → All coordinates are relative to body center")
    print("\n3. Negative values indicate positions left/above center")
    print("   → Positive values indicate positions right/below center")
    print("\n4. Values > 1.0 mean distance greater than shoulder width")
    print("   → Extended arms, raised hands, etc.")
    print("\n5. This normalization is SCALE and POSITION invariant")
    print("   → Same sign looks identical regardless of person size or frame position")
    print("="*70)


if __name__ == "__main__":
    main()
