import os
import shutil
import random

datapath = "D:\Adam\Sekrispi\dataset"
client1_data = "D:\Adam\Sekrispi\client1\dataset"
client2_data = "D:\Adam\Sekrispi\client2\dataset"


for class_name in os.listdir(datapath):
    class_file = os.path.join(datapath, class_name)
    images = os.listdir(class_file)
    random.shuffle(images)

    split_index = int(len(images) * 0.5)
    client1 = images[:split_index]
    client2 = images[split_index:]

    for image_name in client1:
        dst_file = os.path.join(client1_data,class_name)
        os.makedirs(dst_file, exist_ok = True)
        shutil.copy(os.path.join(class_file,image_name), os.path.join(dst_file,image_name))

    for image_name in client2:
        dst_file = os.path.join(client2_data,class_name)
        os.makedirs(dst_file, exist_ok = True)
        shutil.copy(os.path.join(class_file, image_name), os.path.join(dst_file,image_name))

print("success")
