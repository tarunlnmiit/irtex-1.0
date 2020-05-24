# simple sanity checks with a toy dataset
# Run the main method to see analysis for toy dataset at /toy_data,
# no extra setup is needed

import numpy as np
import cv2
import os
from tqdm import tqdm
from numpy.linalg import norm
from  CLDescriptor import  CLDescriptor
import argparse
import traceback

#Cosine
def get_similarity(descriptor1, descriptor2):
    cos_sim = np.dot(descriptor1, descriptor2) / \
        (norm(descriptor1)*norm(descriptor2))
    return cos_sim


#Euclidean
def get_similarity_euclidean(descriptor1, descriptor2):
    descriptor1=descriptor1.reshape(-1,64)
    descriptor2=descriptor2.reshape(-1,64)
    dist=0
    sum=0
    for i,layer in enumerate(descriptor1):
        dist=np.linalg.norm(descriptor1[i] - descriptor2[i])
        sum+= (1/(1+dist))
        #print(i, (1/(1+dist)))


    return  sum/3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLD Extractor Test')
    parser.add_argument('--test_image', help='path to test image. defaults to first image in test folder if empty')
    args = parser.parse_args()


    path='toy_data/'
    feature_list=[]
    computer = CLDescriptor()

    test_image=args.test_image

    if(test_image):
        try:
            img=cv2.imread(test_image)
            descriptor=computer.compute(img)
            feature_list.append(['test_image', descriptor, [0]])
        except Exception as e:
            print(traceback.print_exc())
            print('failed to load image, using default test...')
            pass



    for file in tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path, file))
        descriptor = computer.compute(img)
        feature_list.append([file, descriptor,[0]])


    query = feature_list[0]

    count = 0

    print('comparison with ', query[0])
    for i in feature_list:
        x1 = i[1]
        x2 = query[1]
        #cosine similarity
        feature_list[count][2] = get_similarity(x1, x2)

        print(i[0],'cosine','\t','\t',i[2])
        #Euclidean
        feature_list[count][2] = get_similarity_euclidean(x1, x2)

        print(i[0],'euclid','\t','\t',i[2])
        count+=1


