### Downloads comic images and their corresponding labels from Danbooru; stores labels in .json file

from constants import *

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