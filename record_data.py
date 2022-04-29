import pandas as pd
import os

path = './Cars logo/MLpicture/'
brands = []
images = []
b = []
for name in os.listdir(path):
    b.append(name)

# Build train dataset's info
for n in b:
    p  = path + n + '/'
    for i in os.listdir(p):
        place = p + i
        brands.append(n)
        images.append(place)
data = {'brand': brands, 'image': images}
df = pd.DataFrame(data)

print(df)
print(len(df))
df.to_csv('./Cars logo/info.csv')


# Build test dataset's info
brands, images = [], []
test_path = './Cars logo/test/'
for i in os.listdir(test_path):
    p = test_path + i
    brand = i[:-8]
    brands.append(brand)
    images.append(p)
test_data = {'brand': brands, 'image': images}
df = pd.DataFrame(test_data)

print(df)
print(len(df))
df.to_csv('./Cars logo/test_info.csv')
