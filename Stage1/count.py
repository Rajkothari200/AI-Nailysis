import os

# =====================================
# CHANGE THIS IF NEEDED
# =====================================
dataset_root = r"D:\AI Nail Analysis\AI_Nailysis_stage1"

valid_ext = (".jpg", ".jpeg", ".png")

def count_images(folder_path):
    return len([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    ])

def analyze_split(split_name):
    print(f"\n========== {split_name.upper()} ==========")
    split_path = os.path.join(dataset_root, split_name)

    if not os.path.exists(split_path):
        print("Folder not found.")
        return

    total_split = 0

    for cls in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, cls)

        if os.path.isdir(class_path):
            count = count_images(class_path)
            total_split += count
            print(f"{cls:25s} → {count}")

    print(f"\nTotal {split_name} images: {total_split}")

def main():
    print("📊 DATASET ANALYSIS REPORT")
    print("=" * 40)

    for split in ["train", "val", "test"]:
        analyze_split(split)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()