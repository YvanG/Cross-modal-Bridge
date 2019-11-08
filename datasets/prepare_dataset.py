import glob
import numpy as np
import cv2


directory = 'D:/Data/CUHK/photos/'
directory_path = directory+'*.jpg'
# directory = 'D:/Data/FERET/colorferet/Imgs/'
# directory_path = directory+'*.png'
# --------------------------------------------------
# directory = 'D:/Data/LFW/lfwcrop_grey/faces/'
# directory_path = directory+'*.pgm'
# ---------------------------------------------------
# directory = 'D:/Data/CASIA-WebFace_specom/0000114/'
# directory_path = directory+'*.jpg'
# ----------------------------------------------------
files = glob.glob(directory_path)
output_path = 'D:/Python_scripts/pytorch-CycleGAN-and-pix2pix-master/datasets/feret_sketch_highres/test/'
poc = 0

for f in files:
    poc +=1
    if poc >100:
        break
    image = cv2.imread(f,0)
    cv2.destroyAllWindows()
    # image = cv2.resize(image, (372,426))
    im_AB = np.concatenate([image, image], 1)
    # cv2.imshow('image', im_AB)
    # cv2.waitKey(0)
    # if poc <10:
    #     cv2.imwrite(output_path+'0000'+str(poc)+'.png', im_AB)
    # elif poc < 100:
    #     cv2.imwrite(output_path + '000' + str(poc) + '.png', im_AB)
    # elif poc < 1000:
    #     cv2.imwrite(output_path + '00' + str(poc) + '.png', im_AB)
    # elif poc < 10000:
    #     cv2.imwrite(output_path + '0' + str(poc) + '.png', im_AB)