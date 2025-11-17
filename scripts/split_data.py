# Script to split top 10 classes into new dataset structure
import os
import csv
import shutil

TOP_CLASSES = [
	'DOG1', 'HURDLE/TRIP1', 'BREAKFAST1', 'DARK1', 'DEMAND1',
	'BITE1', 'WHATFOR1', 'DECIDE1', 'ROCKINGCHAIR1', 'DEAF1'
]

# Mapping from original class names to cleaned names
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

SPLITS = ['train', 'test', 'val']
SPLITS_DIR = '../splits'
VIDEOS_DIR = '../videos'
DATASET_DIR = '../dataset'

def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def get_split_files():
	return {split: os.path.join(SPLITS_DIR, f'{split}.csv') for split in SPLITS}

def collect_class_files(split_file):
	class_files = {cls: [] for cls in TOP_CLASSES}
	with open(split_file, newline='', encoding='utf-8') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			gloss = row['Gloss']
			if gloss in TOP_CLASSES:
				class_files[gloss].append(row['Video file'])
	return class_files

def copy_files(class_files, split):
	for cls, files in class_files.items():
		# Use cleaned class name (fallback to original if not in mapping)
		cleaned_cls = CLASS_NAME_MAPPING.get(cls, cls)
		if cleaned_cls is None:
			cleaned_cls = cls
		out_dir = os.path.join(DATASET_DIR, split, cleaned_cls)
		ensure_dir(out_dir)
		for fname in files:
			src = os.path.join(VIDEOS_DIR, fname)
			dst = os.path.join(out_dir, fname)
			if os.path.exists(src):
				shutil.copy2(src, dst)
			else:
				print(f'Warning: {src} does not exist.')

def main():
	ensure_dir(DATASET_DIR)
	for split, split_file in get_split_files().items():
		class_files = collect_class_files(split_file)
		copy_files(class_files, split)

if __name__ == '__main__':
	main()
