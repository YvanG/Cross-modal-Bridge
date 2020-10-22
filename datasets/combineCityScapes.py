import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_RGB',  help='input directory for image A', type=str, default='../dataset/CityScapesRGB')
parser.add_argument('--folder_ann', help='input directory for image B', type=str, default='../dataset/CityScapesAnn')
parser.add_argument('--output_folder', help='output directory for combined images', type=str, default='dataset/CityScapesCombined')

args = parser.parse_args()



