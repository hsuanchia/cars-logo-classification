from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50V2, DenseNet121
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json, cv2

# Configs
epochs = 10
dropout_rate = 0.1
csv_path = './Cars logo/info.csv'
input_size = 224

# Precessing image
def preprocessing_img(path):
    # 等比例放大縮小成統一尺寸
    img = image.load_img(path) 
    x = image.img_to_array(img)
    x /= 255.0 # regularization
    return x
# Import data
data = pd.read_csv(csv_path)

class_name = data['brand'].unique()
classes = len(data['brand'].unique())
class_to_num = {}
num_to_class = {}
for i in range(0,classes):
    class_to_num[class_name[i]] = i
    num_to_class[i] = class_name[i]
# print(class_to_num)
# print(num_to_class)
op = open("class_to_num.json","w")
json.dump(class_to_num,op)
op = open("num_to_class.json","w")
json.dump(num_to_class,op)

train_x,train_y = [], []
print('Total class: ',classes)
print('Total data: ',len(data))


print("Import and preprocessing images")
for i in tqdm(range(0,len(data))):
    p = data['image'][i]
    # s = plt.imread(p)
    # plt.imshow(s)
    # plt.show()
    train_x.append(preprocessing_img(p))
    train_y.append(class_to_num[data['brand'][i]])

# Shuffle data
xy = list(zip(train_x,train_y))
np.random.shuffle(xy)
train_x, train_y = zip(*xy)

train_x = np.array(train_x)
train_y = np.array(train_y)

# one-hot
train_y = tf.keras.utils.to_categorical(train_y,num_classes=classes)


# print(train_x)
# print(train_y)

'''
# Neural Network - Modify from Alexnet (Overfitting)
model = Sequential()

model.add(Conv2D(16,input_shape=(input_size,input_size,3),kernel_size=(7,7),strides=4,padding='same',activation='ReLU',name='block1_conv1'))
model.add(BatchNormalization(name='bn1'))
model.add(Conv2D(32,kernel_size=(5,5),strides=2,padding='same',activation='ReLU',name='block1_conv2'))
model.add(BatchNormalization(name='bn2'))
model.add(Conv2D(64,kernel_size=(3,3),strides=1,padding='same',activation='ReLU',name='block1_conv3'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,name='block1_maxpooling1'))

model.add(Conv2D(32,kernel_size=(5,5),strides=1,padding='same',activation='ReLU',name='block2_conv1'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,name='block2_maxpooling1'))

model.add(Conv2D(64,kernel_size=(3,3),strides=1,padding='same',activation='ReLU',name='block3_conv1'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,name='block3_maxpooling1'))

model.add(Flatten(name='Flatten'))

model.add(Dense(256,activation='relu',name='D1'))
model.add(Dropout(dropout_rate,name='dropout_1'))
model.add(Dense(256,activation='relu',name='D2'))
model.add(Dropout(dropout_rate,name='dropout_2'))
model.add(Dense(classes,activation='softmax',name='output'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(train_x,train_y,epochs=epochs,batch_size=16,validation_split=0.2,verbose=1)

model.save('./Models/cnn_2.h5')
'''


# Neural network - transfer learning from VGG16, densenet, resnet
vgg16 = VGG16(input_shape=(input_size,input_size,3),weights='imagenet',include_top=False)
densenet = DenseNet121(input_shape=(input_size,input_size,3),weights='imagenet',include_top=False)
resnet = ResNet50V2(input_shape=(input_size,input_size,3),weights='imagenet',include_top=False)

for l in densenet.layers:
    l.trainable = False

x = Flatten(name='flatten')(resnet.output)
pred = Dense(classes,activation='softmax',name='output')(x)

transfer_model = Model(inputs=resnet.input,outputs=pred)

transfer_model.summary()

transfer_model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])

history = transfer_model.fit(train_x,train_y,epochs=epochs,batch_size=16,validation_split=0.2,verbose=1)

transfer_model.save('./Models/resnet_3.h5')


def show_train_history(model, train_acc,val_acc):
    plt.plot(model.history[train_acc])
    plt.plot(model.history[val_acc])
    plt.title('Train History')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

show_train_history(history,'accuracy','val_accuracy')
show_train_history(history,'loss','val_loss')