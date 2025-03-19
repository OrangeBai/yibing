import os
import pandas as pd
import numpy as np
import arcticdb as adb
import logging
from tqdm import tqdm


ARCTIC_PATH = "/mnt/f/Yibing/Data/ADB"
PROJECTS = ['ipo', 'lcr']

def get_adb(project):
    assert project.lower() in PROJECTS
    arctic_loc = "lmdb://" + os.path.join(ARCTIC_PATH, project.lower())
    return adb.Arctic(arctic_loc)

def get_lib(project, lib):
    return get_adb(project).get_library(lib)

def list_lib(project):
    return get_adb(project).list_libraries()

def create_lib(project, lib):
    adb = get_adb(project)
    adb.create_library(lib)