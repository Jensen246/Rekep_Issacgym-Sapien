import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import pickle
from sapien_gen_data_path import SapienDataGenerator

stage_result_list = pickle.load(open(os.path.join('test', 'stage_result_list.pkl'), 'rb'))

sapien_data_generator = SapienDataGenerator()
sapien_data_generator.plan_and_execute_path_from_stage_result_list(stage_result_list)
