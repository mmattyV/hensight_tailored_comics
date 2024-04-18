import os
from constants import *
import torch
import multiprocessing
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import os
import requests
import json 
from tqdm import tqdm
import time
from torch.autograd import Variable
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import numpy as np
import matplotlib.pyplot as plt

#TOTAL_POSTS = 7400200 # Total number of posts on Danbooru
IMG_PER_BATCH = 200 # Read limit from Danbooru
TAGS = ['action', 'looking_at_another', 'romance', 'sad', 'crying', 'angry', 'scared', 'surprised', 'fighting', 'chase', 'talking', 'couple'] # Tags for our categories
#STOP_ID = 7400200
NUM_TAGS = len(TAGS)
## TEMP ##
TOTAL_POSTS = 3727400
JSON_FILE = 'comic_labels.json'
TRAIN_JSON = 'comic_labels_train.json'
VAL_JSON = 'comic_labels_val.json'
GPU_MODE = 0
MPS_MODE = 1
CUDA_DEVICE = 0
NUM_EPOCHS = 30
BASE_LR = 0.001
DECAY_WEIGHT = 0.1 
EPOCH_DECAY = 30 
BATCH_SIZE = 10
IMAGES_DIR = '../comic_images'

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
