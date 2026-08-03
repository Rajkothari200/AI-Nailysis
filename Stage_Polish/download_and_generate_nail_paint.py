import os
import shutil
import random
import re
import urllib.request
from PIL import Image, ImageDraw, ImageFilter

# Config
dataset_root = "Stage_Polish"
stage1_root = "Stage1"

splits = {
    "train": {"natural_source_healthy": "Stage1/train/healthy", "natural_source_disease": "Stage1/train/disease", "count": 600},
    "val": {"natural_source_healthy": "Stage1/val/healthy", "natural_source_disease": "Stage1/val/disease", "count": 150},
    "test": {"natural_source_healthy": "Stage1/test/healthy", "natural_source_disease": "Stage1/test/disease", "count": 150}
}

# Create dirs
for split in splits.keys():
    os.makedirs(os.path.join(dataset_root, split, "natural"), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, split, "polish"), exist_ok=True)

print("Directories initialized.")

# Populate natural classes by copying from Stage 1
def populate_natural():
    print("\n--- Populating Natural Nails ---")
    for split, config in splits.items():
        dest_dir = os.path.join(dataset_root, split, "natural")
        
        # Clear existing
        for f in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, f))
            
        h_src = config["natural_source_healthy"]
        d_src = config["natural_source_disease"]
        
        h_imgs = [os.path.join(h_src, f) for f in os.listdir(h_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        d_imgs = [os.path.join(d_src, f) for f in os.listdir(d_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        target_count = config["count"]
        half_count = target_count // 2
        
        sampled_h = random.sample(h_imgs, min(half_count, len(h_imgs)))
        sampled_d = random.sample(d_imgs, min(half_count, len(d_imgs)))
        
        total_copied = 0
        for src_path in sampled_h + sampled_d:
            fname = os.path.basename(src_path)
            shutil.copy(src_path, os.path.join(dest_dir, f"nat_{total_copied}_{fname}"))
            total_copied += 1
            
        print(f"Copied {total_copied} images to {dest_dir}")

# Apply synthetic nail paint overlays
def apply_synthetic_polish(img_path, dest_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        
        # Create overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Color palette for nail polish
        colors = [
            (255, 0, 0, 240),      # Red
            (255, 20, 147, 240),   # Deep Pink
            (199, 21, 133, 240),   # Medium Violet Red
            (0, 0, 255, 240),      # Blue
            (0, 0, 0, 255),        # Black
            (255, 255, 255, 255),  # White
            (128, 0, 128, 240),    # Purple
            (0, 255, 255, 240),    # Cyan/Neon Blue
            (255, 165, 0, 240),    # Orange
            (255, 215, 0, 255),    # Gold
            (144, 238, 144, 240),  # Light Green
        ]
        
        base_color = random.choice(colors)
        
        # Draw central nail shape (ellipse)
        # Assuming nail is centered
        cx, cy = w // 2, h // 2
        rx = random.randint(w // 6, w // 4)
        ry = random.randint(h // 5, h // 3)
        
        # Add slight rotation / angle
        angle = random.randint(-15, 15)
        
        # Create mask for ellipse
        mask = Image.new("L", img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        mask = mask.rotate(angle, resample=Image.BICUBIC)
        
        # Draw base polish color on overlay
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=base_color)
        
        # Decide if we want nail art (lines, stripes, dots)
        if random.random() > 0.5:
            art_colors = [(255, 255, 255, 255), (0, 0, 0, 255), (255, 215, 0, 255), (255, 20, 147, 255)]
            art_color = random.choice(art_colors)
            while art_color == base_color:
                art_color = random.choice(art_colors)
                
            art_type = random.choice(["stripes", "dots", "french_tip"])
            
            if art_type == "stripes":
                # Draw diagonal lines
                for offset in range(-ry, ry, random.randint(15, 30)):
                    draw.line([cx - rx, cy + offset, cx + rx, cy + offset + random.randint(-10, 10)], fill=art_color, width=random.randint(2, 6))
            elif art_type == "dots":
                # Draw random dots
                for _ in range(random.randint(5, 15)):
                    dx = random.randint(cx - rx + 5, cx + rx - 5)
                    dy = random.randint(cy - ry + 5, cy + ry - 5)
                    dr = random.randint(2, 5)
                    draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=art_color)
            elif art_type == "french_tip":
                # Tip color at the end of the nail (top portion)
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy - ry + (ry // 2)], fill=art_color)
                
        # Rotate overlay to match mask
        overlay = overlay.rotate(angle, resample=Image.BICUBIC)
        
        # Apply slight blur to mask edges for natural look
        mask = mask.filter(ImageFilter.GaussianBlur(1))
        
        # Composite overlay on original image using mask
        final_img = Image.composite(overlay, img, mask)
        final_img.save(dest_path, "JPEG")

# Download real images from Unsplash
def download_real_polish():
    print("\n--- Downloading Real Nail Polish & Art Images ---")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    queries = ["nail-polish", "nail-art", "manicure"]
    download_urls = []
    
    for q in queries:
        url = f"https://unsplash.com/s/photos/{q}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                # Find Unsplash photo links
                matches = re.findall(r'https://images.unsplash.com/photo-[a-zA-Z0-9\-_]+', html)
                for match in matches:
                    # Append parameters to size it down
                    full_url = f"{match}?w=400&h=400&fit=crop"
                    if full_url not in download_urls:
                        download_urls.append(full_url)
        except Exception as e:
            print(f"Error scraping query '{q}': {e}")
            
    print(f"Found {len(download_urls)} candidate real-world image URLs.")
    
    # We will download up to 60 real polish images
    # We will distribute them across train (40), val (10), test (10)
    random.shuffle(download_urls)
    
    download_distribution = {
        "train": download_urls[:40],
        "val": download_urls[40:50],
        "test": download_urls[50:60]
    }
    
    for split, urls in download_distribution.items():
        dest_dir = os.path.join(dataset_root, split, "polish")
        downloaded = 0
        for i, url in enumerate(urls):
            try:
                dest_path = os.path.join(dest_dir, f"real_{i}.jpg")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                    out_file.write(response.read())
                downloaded += 1
            except Exception as e:
                pass
        print(f"Downloaded {downloaded} real-world polish images for {split}")

def generate_synthetic_dataset():
    print("\n--- Generating Synthetic Polish Images ---")
    for split in splits.keys():
        natural_dir = os.path.join(dataset_root, split, "natural")
        polish_dir = os.path.join(dataset_root, split, "polish")
        
        natural_imgs = [os.path.join(natural_dir, f) for f in os.listdir(natural_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        synthetic_count = 0
        for img_path in natural_imgs:
            fname = os.path.basename(img_path)
            dest_path = os.path.join(polish_dir, f"synth_{fname}")
            try:
                apply_synthetic_polish(img_path, dest_path)
                synthetic_count += 1
            except Exception as e:
                print(f"Error synthesizing {fname}: {e}")
                
        print(f"Generated {synthetic_count} synthetic polish images for {split}")

def main():
    populate_natural()
    download_real_polish()
    generate_synthetic_dataset()
    print("\n📊 Dataset creation completed successfully!")
    
if __name__ == "__main__":
    main()
