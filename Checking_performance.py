import glob, os, hashlib
from PIL import Image, UnidentifiedImageError
import imagehash

# --- change this to your dataset root ---
dataset_root = r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Training_data_streaks\dataset"

splits = ["train", "val", "test"]  # adjust if no test split
img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# --- collect images from each split ---
split_imgs = {}
for s in splits:
    img_dir = os.path.join(dataset_root, s, "images")
    if os.path.exists(img_dir):
        split_imgs[s] = glob.glob(os.path.join(img_dir, "**/*.*"), recursive=True)
        split_imgs[s] = [p for p in split_imgs[s] if p.lower().endswith(img_exts)]
        print(f"{s}: {len(split_imgs[s])} images")
    else:
        split_imgs[s] = []
        print(f"{s}: 0 images (folder not found)")


# --- exact duplicate check ---
def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        h.update(f.read())
    return h.hexdigest()

hashes, dups = {}, []
for s in splits:
    for p in split_imgs[s]:
        h = file_hash(p)
        if h in hashes:
            dups.append((hashes[h], p))
        else:
            hashes[h] = p

print(f"\nExact duplicates found: {len(dups)}")
for a,b in dups[:10]:
    print("DUP:", a, "<->", b)

# --- near duplicate check (phash) ---
hashes_phash, near_dups = {}, []
for s in splits:
    for p in split_imgs[s]:
        try:
            with Image.open(p) as img:
                h = str(imagehash.phash(img))
            if h in hashes_phash:
                near_dups.append((hashes_phash[h], p))
            else:
                hashes_phash[h] = p
        except UnidentifiedImageError:
            print("Skipping non-image file:", p)
        except Exception as e:
            print("Error reading", p, e)

print(f"\nNear-duplicates found: {len(near_dups)}")
for a,b in near_dups[:10]:
    print("NEAR DUP:", a, "<->", b)
