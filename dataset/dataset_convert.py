# データセット用にピックアップ → custom_pretrain.json として保存
import os
import json

dataset_dir = "./images"
json_dataset_path = "cc3m_pretrain_595k_ja.json"
output_json = "custom_pretrain.json"

def custom_dataset(dataset_dir, json_dataset_path, output_json):
    with open(json_dataset_path) as f:
        dataset = json.load(f)

    checked = 0
    pickup = []
    for data in dataset:
        if "image" in data.keys():
            image_path = data["image"]
            if os.path.isfile(os.path.join(dataset_dir, image_path)):
                checked += 1
                pickup.append(data)
    with open(output_json, "w") as f:
        json.dump(pickup, f, indent=4)
    print(f"result: {checked}/{len(dataset)}")

# run
custom_dataset(dataset_dir, json_dataset_path, output_json)