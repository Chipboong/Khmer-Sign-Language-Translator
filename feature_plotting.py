# plot_parquet_pixels.py
"""
Plot landmarks as pixel coordinates from your parquet file.

Defaults:
  PARQUET_PATH: your provided file path
  WIDTH x HEIGHT: default pixel frame size (change to match your video: e.g., 1280x720 or 1920x1080)

Behavior:
  - If x.max() <= 1.01 and y.max() <= 1.01 we assume normalized coords in [0,1] and scale them to pixels.
  - If your coordinates are already in pixels the script will use them directly.
  - Optionally overlay on an image if you set OVERLAY_IMAGE to a valid image path.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

# ------------------ CONFIG ------------------
PARQUET_PATH = r"D:\VS Code\CV\American Sign Language Translator\dataset/train_landmark_files/37779/67916304.parquet"

# change to your frame size (pixels) if coordinates are normalized
WIDTH = 1.5   # <-- change if your video frame width is different
HEIGHT = 1.5   # <-- change if your video frame height is different

# Optional: overlay landmarks on an image (frame). Set to None to skip.
OVERLAY_IMAGE = None  # e.g. r"D:\path\to\frame_0001.png" or None
# Overlay alpha (transparency)
OVERLAY_ALPHA = 0.6

# Which frame to plot by default (set None to plot the first frame found)
PLOT_FRAME_INDEX = None

# ------------------------------------------------

def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    required = {"frame","landmark_index","x","y"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {required}, got {df.columns.tolist()}")
    return df

def to_pixel_coords(frame_df, width, height):
    """
    Input frame_df with columns x,y (and optionally z).
    If values look normalized (max <= 1.01), convert to pixel coords:
      x_px = x * width
      y_px = y * height
    Returns DataFrame with x_px, y_px, and preserved other cols.
    """
    xs = frame_df['x'].values
    ys = frame_df['y'].values
    max_x = np.nanmax(np.abs(xs))
    max_y = np.nanmax(np.abs(ys))
    df_out = frame_df.copy()
    # Heuristic: if coordinates in [-1,1] or [0,1], treat as normalized
    if max_x <= 1.01 and max_y <= 1.01:
        df_out['x_px'] = (df_out['x'].astype(float) * width).astype(float)
        df_out['y_px'] = (df_out['y'].astype(float) * height).astype(float)
        print(f"Detected normalized coordinates; scaled to pixels using width={width}, height={height}")
    else:
        # assume already in pixels
        df_out['x_px'] = df_out['x'].astype(float)
        df_out['y_px'] = df_out['y'].astype(float)
        print("Detected pixel coordinates (no scaling applied).")
    return df_out

def plot_single_frame(frame_df_px, frame_id=None, overlay_image=None, title=None, annotate=True, save_path=None):
    """
    Plot landmarks in pixel coords. Origin top-left; y increases downwards.
    frame_df_px must have x_px and y_px columns.
    """
    x = frame_df_px['x_px'].values
    y = frame_df_px['y_px'].values

    fig, ax = plt.subplots(figsize=(6, 8))
    # overlay image if provided
    if overlay_image is not None and os.path.exists(overlay_image):
        img = Image.open(overlay_image).convert("RGB")
        ax.imshow(img, alpha=OVERLAY_ALPHA, extent=[0, img.width, img.height, 0])
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
    else:
        ax.set_xlim(-0.5, WIDTH)
        ax.set_ylim(HEIGHT, 0)

    ax.scatter(x, y, s=60, edgecolors='k')
    if annotate:
        for i, (xi, yi, li) in enumerate(zip(x, y, frame_df_px['landmark_index'])):
            ax.text(xi + 4, yi + 4, str(int(li)), fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    if title is None:
        title = f"Frame {frame_id}" if frame_id is not None else "Frame"
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()

def animate_landmarks(df, width, height, overlay_image=None, frames_to_plot=None, interval=50, skip=1):
    """
    Animate landmarks across frames.
    frames_to_plot: list of frame numbers to visualize (if None use all)
    """
    frames = sorted(df['frame'].unique())
    if frames_to_plot is None:
        frames_to_plot = frames
    else:
        frames_to_plot = [f for f in frames if f in set(frames_to_plot)]

    # precompute per-frame pixel coords
    frame_dfs = []
    for f in frames_to_plot:
        fr = df[df['frame']==f].sort_values('landmark_index')
        fr_px = to_pixel_coords(fr, width, height)
        frame_dfs.append((f, fr_px))

    # Setup plot
    fig, ax = plt.subplots(figsize=(6, 8))
    if overlay_image and os.path.exists(overlay_image):
        img = Image.open(overlay_image).convert("RGB")
        ax.imshow(img, alpha=OVERLAY_ALPHA, extent=[0, img.width, img.height, 0])
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
    else:
        ax.set_xlim(-0.5, width)
        ax.set_ylim(height, 0)
    scat = ax.scatter([], [], s=60, edgecolors='k')
    title = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center")

    def init():
        scat.set_offsets(np.empty((0,2)))
        title.set_text("")
        return scat, title

    def update(i):
        f, fr_px = frame_dfs[i]
        coords = np.column_stack([fr_px['x_px'].values, fr_px['y_px'].values])
        scat.set_offsets(coords)
        title.set_text(f"Frame {f} ({i+1}/{len(frame_dfs)})")
        return scat, title

    ani = animation.FuncAnimation(fig, update, frames=len(frame_dfs), init_func=init, interval=interval, blit=False)
    plt.show()
    return ani

def main():
    df = load_df(PARQUET_PATH)
    print("Loaded rows:", len(df), "unique frames:", df['frame'].nunique())
    # choose frame to plot
    if PLOT_FRAME_INDEX is None:
        chosen_frame = sorted(df['frame'].unique())[0]
    else:
        chosen_frame = PLOT_FRAME_INDEX

    fr = df[df['frame'] == chosen_frame].sort_values('landmark_index')
    if fr.empty:
        raise ValueError(f"No rows found for frame {chosen_frame}")

    fr_px = to_pixel_coords(fr, WIDTH, HEIGHT)
    # Plot single frame
    plot_single_frame(fr_px, frame_id=chosen_frame, overlay_image=OVERLAY_IMAGE, annotate=True, save_path=None)

    # Optionally animate whole sequence (comment out if not desired)
    do_animate = True
    if do_animate:
        print("Animating frames (this may take a moment)...")
        animate_landmarks(df, WIDTH, HEIGHT, overlay_image=OVERLAY_IMAGE, interval=50, skip=1)

if __name__ == "__main__":
    main()
