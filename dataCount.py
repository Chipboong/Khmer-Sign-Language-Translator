import json

with open('New_MSL_Dataset/MSL_train.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
print(len(data))