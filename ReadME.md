# ASL Citizen - Real-time Sign Language Recognition

A real-time American Sign Language (ASL) recognition system using **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** and **MediaPipe** for keypoint extraction.

## 🎯 Features

- **Real-time Sign Detection**: Automatic hand presence detection triggers sign capture
- **High Accuracy**: ST-GCN model trained on ASL sign language dataset
- **Smart Activity Detection**: Captures signs only when hands are present
- **Cross-platform**: Works on Windows and Linux
- **Easy to Use**: Simple command-line interface

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- MediaPipe
- NumPy

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ASL_Citizen.git
cd ASL_Citizen
```

2. **Install dependencies**
```bash
pip install tensorflow opencv-python mediapipe numpy tqdm
```

3. **Verify installation**
```bash
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('All dependencies installed!')"
```

## 📁 Project Structure

```
ASL_Citizen/
├── src/
│   ├── prepare_stgcn_data.py    # Data preprocessing & keypoint extraction
│   ├── train_stgcn.py            # ST-GCN model training
│   ├── predict_stgcn.py          # Prediction utilities
│   └── models/
│       └── stgcn_tf.py           # ST-GCN model architecture
├── realtime_predict.py           # Real-time webcam prediction
├── predict_video.py              # Single video file prediction
├── scripts/
│   └── split_data.py             # Dataset splitting utility
├── splits/                       # Train/val/test split files
└── README.md
```

## 🎬 Usage

### 1. Prepare Dataset

Organize your video dataset:
```
dataset/
├── train/
│   ├── HELLO/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   ├── THANKS/
│   └── ...
├── val/
└── test/
```

Preprocess videos and extract keypoints:
```bash
python src/prepare_stgcn_data.py
```

**What it does:**
- Extracts 543 MediaPipe landmarks from each video frame
- Selects 27 key body/hand keypoints for ST-GCN
- Normalizes keypoints using shoulder-based scaling
- Resamples all videos to 128 frames
- Saves processed data as `.npy` files

### 2. Train Model

Train the ST-GCN model:
```bash
python src/train_stgcn.py
```

**Training features:**
- Spatial-Temporal Graph Convolutional Network
- Early stopping and model checkpointing
- Learning rate scheduling
- Validation monitoring
- Saves best model weights

### 3. Real-time Prediction

Run real-time sign language recognition with your webcam:
```bash
python realtime_predict.py
```

**With custom settings:**
```bash
python realtime_predict.py \
  --weights models_stgcn/best_model_weights.weights.h5 \
  --config models_stgcn/config.json \
  --confidence_threshold 0.7 \
  --idle_frames 15 \
  --min_frames 10 \
  --max_frames 200
```

**How it works:**
- 🟢 **Hands detected** → Starts capturing keypoints
- 🔴 **Signing** → Buffers frames during active signing
- ⚪ **Idle** → After 15 frames without hands, processes and predicts
- ✅ **Result** → Displays prediction with confidence score

**Controls:**
- `q` - Quit
- `r` - Reset buffer and clear prediction

### 4. Predict from Video File

Process a single video file:
```bash
python predict_video.py --video path/to/sign_video.mp4
```

**Example:**
```bash
python predict_video.py \
  --video dataset/test/BITE/sample.mp4 \
  --weights models_stgcn/best_model_weights.weights.h5 \
  --config models_stgcn/config.json \
  --target_frames 128
```

## 🧠 Model Architecture

**ST-GCN (Spatial-Temporal Graph Convolutional Network)**

- **Input**: (128 frames, 27 keypoints, 2 coordinates)
- **Keypoints**: 11 pose + 8 left hand + 8 right hand
- **Architecture**:
  - Spatial graph convolutions capture body pose relationships
  - Temporal convolutions model motion over time
  - Multi-layer GCN with residual connections
  - Dropout for regularization

**Training Details:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Data Augmentation: Random scaling, rotation, translation
- Early Stopping: Patience 20 epochs
- Learning Rate: ReduceLROnPlateau

## 📊 Dataset Format

**Video Requirements:**
- Format: `.mp4`, `.avi`, `.mov`
- Frame rate: 30 fps (recommended)
- Duration: 1-3 seconds per sign
- Resolution: Any (will be processed by MediaPipe)

**Processed Data:**
- Shape: `(128, 27, 2)` per sample
- 128 frames uniformly sampled
- 27 keypoints (pose + hands)
- 2 channels (x, y coordinates)
- Normalized with shoulder-based scaling

## 🎨 Keypoint Selection

From 543 MediaPipe landmarks → **27 key points**:

- **Pose (11)**: Head, shoulders, elbows, wrists, hips
- **Left Hand (8)**: Key finger joints
- **Right Hand (8)**: Key finger joints

This selection focuses on essential signing features while keeping computational efficiency.

## ⚙️ Configuration

Key parameters in `realtime_predict.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence_threshold` | 0.7 | Minimum confidence to display prediction |
| `--idle_frames` | 10 | Frames without hands before sign ends |
| `--min_frames` | 10 | Minimum frames for valid sign |
| `--max_frames` | 200 | Maximum frames to capture |
| `--target_frames` | 128 | Model input size (resampled) |

## 📈 Performance Tips

1. **Good lighting**: Ensure hands are well-lit and visible
2. **Clear background**: Reduce background clutter
3. **Hand positioning**: Keep hands in camera view
4. **Sign duration**: 1-3 seconds per sign works best
5. **Idle time**: Wait briefly between signs for better detection
