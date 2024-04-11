from constants import *
import os
from PIL import Image
import json
import tqdm
import requests

# Creates directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Returns the tags from TAGS that are in the tag_string of a post; returns False if none of our TAGS are in tag_string
def check_tags_in_tag_string(tag_string, tags):
    tag_list = []
    for tag in tags:
        if tag in tag_string:
            tag_list.append(tag)
    if len(tag_list) == 0:
        return False
    else:
        return tag_list
    
# Copied from custom_hymenoptera_dataset.py
# Checks for valid image files (size and extension)
def is_valid_image_file(filename, max_pixels=178956970):
    # Check file name extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    if os.path.splitext(filename)[1].lower() not in valid_extensions:
        print(f"Invalid image file extension \"{filename}\". Skipping this file...")
        return False
    
    # Temporarily disable the decompression bomb check
    original_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    
    # Verify that image file is intact and check its size
    try:
        with Image.open(filename) as img:
            img.verify()  # Verify if it's an image
            # Restore the original MAX_IMAGE_PIXELS limit
            Image.MAX_IMAGE_PIXELS = original_max_image_pixels
            
            # Check image size without loading the image into memory
            if img.size[0] * img.size[1] > max_pixels:
                print(f"Image {filename} is too large. Skipping this file...")
                return False
            return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file {filename}: {e}")
        # Ensure the MAX_IMAGE_PIXELS limit is restored even if an exception occurs
        Image.MAX_IMAGE_PIXELS = original_max_image_pixels
        return False
    # Ensure the MAX_IMAGE_PIXELS limit is restored in case of any other unexpected exit
    Image.MAX_IMAGE_PIXELS = original_max_image_pixels
    
# Create dir for our comic images
create_dir(IMAGES_DIR)

# Dict for image file name and list of tags
image_label_dict = {}

B = IMG_PER_BATCH

progress_bar = tqdm(total=TOTAL_POSTS)

# Loops over all post id's to download images from posts that are tagged "comic" and contain at least one of our TAGS
# Also creates dict of image file name and associated tags
while B <= TOTAL_POSTS:
    url = f'https://danbooru.donmai.us/posts.json?page=b{B}&page=a{B-IMG_PER_BATCH}&limit={IMG_PER_BATCH}'
    response_pages = requests.get(url)

    response_pages_json = response_pages.json()

    for page in response_pages_json:
        if 'file_url' in page and 'tag_string' in page:
            tag_string = page['tag_string']
            id = page['id']

            scene_tags = check_tags_in_tag_string(tag_string, TAGS)
            if 'comic' in page['tag_string'] and scene_tags:
                file_url = page['file_url']

                image_path = f'{id}.jpg'
        
                if not os.path.exists(os.path.join(IMAGES_DIR, image_path)):
                    response_img = requests.get(file_url)

                    # If post contains relevant tags and has valid image file, save the image with id as name
                    if response_img.status_code == 200:
                        
                        with open(os.path.join(IMAGES_DIR, image_path), 'wb') as file:
                            file.write(response_img.content)
                        
                # Write values to dict
                image_label_dict[image_path] = scene_tags

    B += IMG_PER_BATCH
    progress_bar.update(200)

progress_bar.close()

print(image_label_dict)

# Save dict to .json file
with open(JSON_FILE, 'w') as f: 
     json.dump(image_label_dict, f)

import random

# Break data into train and val .json files
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

    total_items = len(data)
    val_part_size = int(0.2*total_items)
    train_part_size = total_items-val_part_size

    keys = list(data.keys())

    random.shuffle(keys)

    val_part_keys = keys[:val_part_size]
    train_part_keys = keys[val_part_size:]

    val_part_dict = {key: data[key] for key in val_part_keys}
    train_part_dict = {key: data[key] for key in train_part_keys}

    with open(TRAIN_JSON, 'w') as t:
        json.dump(train_part_dict, t)

    with open(VAL_JSON, 'w') as w:
        json.dump(val_part_dict, w)