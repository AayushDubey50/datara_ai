import os

IMG_EXTS = (".jpg", ".jpeg", ".png")

def list_files(base_dir):
    print(f"\n📂 Scanning: {base_dir}")
    count = 0
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                count += 1
                rel = os.path.relpath(os.path.join(root, f), base_dir)
                print(f"  - {rel}")
    print(f"✅ Found {count} images in {base_dir}")

if __name__ == "__main__":
    list_files("dataset/train/images")
    list_files("dataset/train/masks")
    list_files("dataset/val/images")
    list_files("dataset/val/masks")
