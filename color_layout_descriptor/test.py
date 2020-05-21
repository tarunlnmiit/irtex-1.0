import numpy as np
import cv2
import os
from tqdm import tqdm
from numpy.linalg import norm
from  CLDescriptor import  CLDescriptor

def get_similarity(descriptor1, descriptor2):
    cos_sim = np.dot(descriptor1, descriptor2) / \
        (norm(descriptor1)*norm(descriptor2))
    return cos_sim

if __name__ == "__main__":
    path='toy_data/'
    feature_list=[]
    for file in tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path, file))
        computer = CLDescriptor()
        descriptor = np.around(computer.compute(img), decimals=4)
        feature_list.append([file, descriptor,[0]])


    query=feature_list[0]

    count=0

    print('comparison with ', query[0])
    for i in feature_list:
        x1 = i[1]
        x2 = query[1]
        feature_list[count][2] = get_similarity(x1, x2)
        print(i[0],'\t',i[2])
        count+=1


