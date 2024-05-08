#!/bin/bash

# Execute the first Python script
python ResNet50_train.py

# Execute the second Python script
python EffNetB0_train.py

# Execute the third Python script
python custom_train.py

python vertices_model_train.py
