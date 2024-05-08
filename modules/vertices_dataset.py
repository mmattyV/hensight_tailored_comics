from constants import *
from detect_vertices import *

class VerticesDataset(Dataset):
    def __init__(self, images_dir, json_file, processed_file, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform

        if os.path.exists(processed_file):
            print(f"Loading processed data from {processed_file}")
            self.items = torch.load(processed_file)
        else:
            self.items = self.process_data(images_dir, json_file)
            print(f"Saving processed data to {processed_file}")
            torch.save(self.items, processed_file)

    def check_shrink_tags(self, tags):
        output = []
        for tag in tags:
            if tag in TAGS:
                output.append(tag)

        if len(output) == 0:
            return False, output
        else:
            return True, output

    def process_data(self, images_dir, json_file):
        image_label_dict = {}
        class_counts = {}

        for tag in TAGS:
            class_counts[tag] = 0

        with open(json_file, 'r') as f:
            data = json.load(f)

        progress_bar = tqdm(total=len(data.items()))
        for key, value in data.items():
            has_tag, output = self.check_shrink_tags(value)
            if key in os.listdir(images_dir) and has_tag:
                if is_valid_image_file(os.path.join(self.images_dir, key)) and cv2.imread(os.path.join(self.images_dir, key)) is not None:
                    target = self.hot_encode_target(output)

                    if len(target) == NUM_TAGS:
                        image_label_dict[key] = target
                        for o in output:
                            class_counts[o] += 1

                else:
                    print('Invalid file: ' + key + '. Skipping this file...')
            
            progress_bar.update(1)
        
        progress_bar.close()

        print('Class counts: ', class_counts)

        if (sum(class_counts.values()) > 20000):
            phase = "TRAIN"
        else:
            phase = "VAL"
        print(f"{phase.upper()} SET STATISTICS:")
        total_images = sum(class_counts.values())
        print(f"Total images: {total_images}")
        for class_id, count in class_counts.items():
            print(f"Class {class_id}: {count} images")
        print("\n")

        return list(image_label_dict.items())

    # Hot encode our labels for our targets
    def hot_encode_target(self, tags):
        target = torch.zeros(NUM_TAGS, dtype=torch.float)
        for tag in tags:
            target[TAGS.index(tag)] = 1

        return target

    def __len__(self):
        return len(self.items) // 2
    
    def normalize_flatten_vertices(self, vertices, image_size):
        width, height = image_size
        normalized_vertices = []

        for vert_set in vertices:
            for vert in vert_set:
                # Normalizing by scaling by height and width then centering
                normalized_vertices.append([(vert[0]/width) - 0.5, (vert[1]/height) - 0.5])

        flattened_vertices = [item for sublist in normalized_vertices for item in sublist]
        return np.array(flattened_vertices)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.items[idx][0])

        # Load image to get its dimensions
        image = Image.open(img_path).convert('RGB')
        image_size = image.size

        vertices = detect_vertices(img_path, False)
        if vertices is None or len(vertices) == 0:
            norm_flattened_vertices = []
        else:
            norm_flattened_vertices = self.normalize_flatten_vertices(vertices, image_size)

        if len(norm_flattened_vertices) == 0:
            padded_vertices = np.full(MAX_NUM_VERTICES, -10)
        elif len(norm_flattened_vertices) > MAX_NUM_VERTICES:
            padded_vertices = norm_flattened_vertices[:MAX_NUM_VERTICES]
        else:
            padded_vertices = np.copy(norm_flattened_vertices)
            padded_vertices = np.concatenate((padded_vertices, np.full(MAX_NUM_VERTICES - len(norm_flattened_vertices), -10)))
    
        label = self.items[idx][1]

        if self.transform:
            padded_vertices = self.transform(padded_vertices)
        if self.target_transform:
            label = self.target_transform(label)

        padded_vertices = torch.tensor(padded_vertices, dtype=torch.float)
    
        return padded_vertices, label