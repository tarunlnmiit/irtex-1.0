import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import mahotas
from sklearn.metrics.pairwise import cosine_similarity
from skimage.segmentation import slic, felzenszwalb


def takefirst(elem):
    return elem[0]


class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)


desc = ZernikeMoments(50)
zernike = []
images = []

dataFile = os.getcwd()
path = os.path.join(dataFile, 'toy_dataset')
labels = []
for label in tqdm(os.listdir(path)):
    img_array = cv2.imread(os.path.join(path, label), cv2.IMREAD_GRAYSCALE)
    #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (100, 100))

    #img_array = img_array + np.random.normal(scale=0.2, size=img_array.shape)
    #img_array = img_array.astype('float32')

    #img_array = cv2.filter2D(img_array, -1, kernel)
    #img_array = cv2.bilateralFilter(img_array, 2, 5, 5)
    # invert the image and threshold it
    #img_array = cv2.bitwise_not(img_array)
    moments = desc.describe(img_array)
    zernike.append(moments)
    #print(moments)

    labels.append(label)
    images.append(img_array)
    #plt.figure()
    #img_array = slic(img_array, n_segments=100, sigma=2)
    #plt.imshow(img_array, cmap='gray')  # graph it
    #plt.show()  # display!

imgtest = cv2.imread(os.path.join(path, 'circletest.jpg'), cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_UNCHANGED)
imgtest = cv2.resize(imgtest, (100, 100))
test_moments = desc.describe(imgtest)
test_moments = test_moments.reshape(1, -1)

sim_t = []
sim_t = cosine_similarity(zernike, test_moments)
#for moment in zernike:
#    sim = np.linalg.norm((moment - test_moments), ord=1)
#    sim_t.append(sim)
result = []
for i,j in zip(sim_t, labels):
    result.append([i,j])

result.sort(key=takefirst, reverse=True)

for i in result:
    print(i)

