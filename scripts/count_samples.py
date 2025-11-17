# Script to count top 100 classes with most samples from train.csv
import csv
from collections import Counter

csv_path = '../splits/train.csv'

class_counts = Counter()

with open(csv_path, newline='', encoding='utf-8') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		gloss = row['Gloss']
		class_counts[gloss] += 1

top_100 = class_counts.most_common(100)

print(f"Top 100 classes by sample count in {csv_path}:")
for gloss, count in top_100:
	print(f"{gloss}: {count}")
