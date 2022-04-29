from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tqdm import tqdm
import numpy as np
import pandas as pd
import json, os, cv2
import matplotlib.pyplot as plt

def preprocessing_img(path):
    # 等比例放大縮小成統一尺寸
    img = image.load_img(path, target_size=(224, 224)) 
    x = image.img_to_array(img)
    x /= 255.0 # regularization
    return x

model_path = './Models/vgg66_1.h5'
test_dir = './Cars logo/test/' 

model = load_model(model_path)

with open('num_to_class.json','r') as f:
    num_to_class = json.load(f)
print(num_to_class)

def predict(path,real_ans):
    data = []
    tmp = preprocessing_img(path)
    data.append(tmp)
    data = np.array(data)
    pre = model.predict(data)

    # print(pre[0])
    # print('Real answer: ',real_ans)
    ans = -1
    ans_name = 'unknown'
    # Anomaly detection
    for i in pre[0]:
        if i > 0.8:
            ans = 0
            break
    if ans != -1:
        ans = np.argmax(pre[0])
        ans_name = num_to_class[(str)(ans)]
    # print('Predict answer: ',ans_name)

    if ans_name == real_ans:
        return 1
    else:
        return 0
    '''
    img = cv2.imread(path)
    cv2.namedWindow(path,0)
    cv2.resizeWindow(path,500,500)
    cv2.imshow(path,img)
    cv2.waitKey(0)
    '''

# Predict test data and evaluate test accuracy
test_data = pd.read_csv('./Cars logo/test_info.csv')
correct = 0
for i in tqdm(range(0,len(test_data))):
    correct += predict(test_data['image'][i],test_data['brand'][i])

print("Test accuracy: ", (float)(correct) / (float)(len(test_data)))


# Predict Single image
# predict(path,real_answer)
    