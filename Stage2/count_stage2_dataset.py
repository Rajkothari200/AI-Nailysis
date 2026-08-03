import os

dataset_path = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2"

splits = ["train", "val", "test"]

print("\n📊 STAGE 2 DATASET REPORT")
print("=" * 40)

for split in splits:
    
    split_path = os.path.join(dataset_path, split)
    
    print(f"\n====== {split.upper()} ======")
    
    total_images = 0
    
    for disease in sorted(os.listdir(split_path)):
        
        disease_path = os.path.join(split_path, disease)
        
        if os.path.isdir(disease_path):
            
            count = len([
                file for file in os.listdir(disease_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            
            print(f"{disease:<20} → {count}")
            
            total_images += count
    
    print(f"\nTotal {split} images: {total_images}")