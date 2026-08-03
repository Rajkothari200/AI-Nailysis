import os
import random

root = r"D:\AI Nail Analysis\AI_Nailysis_stage1"
splits = ["train", "val", "test"]

valid_ext = (".jpg", ".jpeg", ".png")

def balance_split(split):
    healthy_path = os.path.join(root, split, "healthy")
    disease_path = os.path.join(root, split, "disease")

    healthy_files = [f for f in os.listdir(healthy_path) if f.lower().endswith(valid_ext)]
    disease_files = [f for f in os.listdir(disease_path) if f.lower().endswith(valid_ext)]

    min_count = min(len(healthy_files), len(disease_files))

    random.shuffle(disease_files)

    # Remove excess disease images
    for file in disease_files[min_count:]:
        os.remove(os.path.join(disease_path, file))

    print(f"{split} balanced to {min_count} per class.")

def main():
    for split in splits:
        balance_split(split)

    print("\n✅ Stage-1 dataset balanced.")

if __name__ == "__main__":
    main()