import os
import sys


current_dir = os.path.dirname(os.path.abspath('__file__'))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)


DATA_PATH = os.path.abspath(os.path.join(project_root, 'data'))

FIRST_SAMPLE_DATASET = os.path.abspath(os.path.join(project_root, 'data/Endo-baza-proba-15-10-2023.xlsx'))

FINAL_DATASET = os.path.abspath(os.path.join(project_root, 'data/Endo-baza-calosc - 05.01.2024-final.xlsx'))

PASSWORD_FILE_PATH = os.path.abspath(os.path.join(project_root, 'data/password.txt'))


CHARTS_PATH = os.path.abspath(os.path.join(project_root, 'charts'))