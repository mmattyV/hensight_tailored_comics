### Deletes images that don't have a tag in TAGS

from constants import *

with open(JSON_FILE, 'r') as f:
    data = json.load(f)

i = 0

progress_bar = tqdm(total=len(os.listdir(IMAGES_DIR)))
for path in os.listdir(IMAGES_DIR):
    if (path not in data) or (not is_valid_image_file(os.path.join(IMAGES_DIR, path))):
        os.remove(os.path.join(IMAGES_DIR, path))
        #print(f'Deleted: {path}')
        i += 1
    progress_bar.update(1)

progress_bar.close()

print(f'Deleted {i} files!')
