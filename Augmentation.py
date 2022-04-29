import cv2,os
from tqdm import tqdm

dir = './Cars logo/MLpicture/'

b = []

for name in os.listdir(dir):
    b.append(name)

# Rotate image
def rotate(path):
    for n in tqdm(b):
        p  = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            img_90 = p + 'aug_90_' + i
            img_180 = p + 'aug_180_' + i
            img_270 = p + 'aug_270_' + i
            s = cv2.imread(img)
            r90 = cv2.rotate(s,cv2.cv2.ROTATE_90_CLOCKWISE)
            r180 = cv2.rotate(s,cv2.cv2.ROTATE_180)
            r270 = cv2.rotate(s,cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(img_90,r90)
            cv2.imwrite(img_180,r180)
            cv2.imwrite(img_270,r270)

# Resize image
def resize(path):
    for n in tqdm(b):
        p = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            s = cv2.imread(img)
            s = cv2.resize(s,(224,224))
            cv2.imwrite(img,s)

# Gray scale
def convert_gray(path):
    for n in tqdm(b):
        p = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            s = cv2.imread(img)
            s = cv2.cvtColor(s,cv2.COLOR_RGB2GRAY)
            cv2.imwrite(img,s)

