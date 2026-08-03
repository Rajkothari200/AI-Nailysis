import os
from PIL import Image
from collections import Counter

dataset_path = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2"

splits = ["train", "val", "test"]

sizes = []

print("\n🔍 ANALYZING IMAGE RESOLUTIONS\n")

for split in splits:
    split_path = os.path.join(dataset_path, split)

    for disease in os.listdir(split_path):
        disease_path = os.path.join(split_path, disease)

        if os.path.isdir(disease_path):

            for file in os.listdir(disease_path):

                if file.lower().endswith((".jpg", ".jpeg", ".png")):

                    img_path = os.path.join(disease_path, file)

                    try:
                        with Image.open(img_path) as img:
                            sizes.append(img.size)
                    except:
                        pass

print("Total images checked:", len(sizes))

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

print("\nMin resolution:", min(widths), "x", min(heights))
print("Max resolution:", max(widths), "x", max(heights))

avg_w = sum(widths) // len(widths)
avg_h = sum(heights) // len(heights)

print("Average resolution:", avg_w, "x", avg_h)

# Most common sizes
counter = Counter(sizes)

print("\nMost common resolutions:")
for size, count in counter.most_common(10):
    print(size, "→", count)