import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_RGB',  help='input directory for image A', type=str, default='../datasets/CityScapesRGB')
parser.add_argument('--folder_ann', help='input directory for image B', type=str, default='../datasets/CityScapesAnn')
parser.add_argument('--output_folder', help='output directory for combined images', type=str, default='../datasets/CityScapesCombined')

args = parser.parse_args()


RGBFolders = os.listdir(os.path.join(args.folder_RGB,'test'))
AnnFolders = os.listdir(os.path.join(args.folder_ann,'test'))

for fileRGB, fileAnn in zip(RGBFolders,AnnFolders):
    imagesRGB = os.listdir(os.path.join(args.folder_RGB,'test', fileRGB))
    for image in imagesRGB:
        deleteLen = len(image.split('_')[-1])
        name = image[:-deleteLen]+'gtFine_color.png'
        im_A = cv2.imread(os.path.join(args.folder_RGB,'test', fileRGB, image))
        im_B = cv2.imread(os.path.join(args.folder_ann, 'test', fileAnn, name))
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(os.path.join(args.output_folder,'test',image[:-deleteLen]+'combined.png'), im_AB)
        print('Image '+str(image[:-deleteLen])+' was finished!')

