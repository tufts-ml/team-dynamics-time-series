
import os
from enum import Enum
from dynagroup.model2a.basketball.model import Model_Type
from dynagroup.model import Model


def save_model_type(model_type: Model_Type, model_dir: str, basename_prefix: str = ""):
    filepath = os.path.join(model_dir, f"{basename_prefix}_model_type_string.txt")
    with open(filepath, "w") as file:
        file.write("marching_band")