import os
import sys


current_dir = os.path.dirname(os.path.abspath('__file__'))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)



DATASET_PATH = os.path.abspath(os.path.join(project_root, 'data/preprocessed_dataset.csv'))
CHARTS_PATH = os.path.abspath(os.path.join(project_root, 'charts'))