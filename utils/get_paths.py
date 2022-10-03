import multiprocessing
from pathlib import Path
import os

local = multiprocessing.cpu_count() == 4

# set the path containing all the input data
dir_path = os.path.dirname(os.path.realpath(__file__))
INPUT_PATH = dir_path + "/../input_data/"  # !!!!! change this for your own code

# make the paths to write the results to
RESULTS_PATH = dir_path + "/../results"
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
Path(RESULTS_PATH + "/figures_table_paper").mkdir(parents=True, exist_ok=True)
FINAL_RESULTS_PATH = RESULTS_PATH + "/figures_table_paper/"
