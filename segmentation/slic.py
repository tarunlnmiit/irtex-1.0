import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score

#dataFile = "C:\\Users\\Administrator\\Desktop\\libin_ovgu\\SoSe20\\IRTEX Project\\venv\\irtex-1.0-retrieval-engine\\dataset\\cifar10"
dataFile = os.path.join(settings.BASE_DIR, 'media', 'cifar10')
path = os.path.join(dataFile)
images = []
labels = []
target_names = []
i = 0

for label in tqdm(os.listdir(path)):
    target_names.append(label)
    i = 0
    for img in tqdm(os.listdir(os.path.join(path, label))):
        img_array = cv2.imread(os.path.join(path, label, img), cv2.IMREAD_UNCHANGED)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (256, 256))

        images.append(img_array)
        labels.append(label)
        i +=1
        #if i == 2:
        #    break

img_seg = []

for img in images:
    # apply SLIC and extract (approximately) the supplied number
    # of segments	ax.imshow(mark_boundaries(image, segments))
    segments = slic(img, n_segments=10, sigma = 2, max_iter=500, convert2lab=True, slic_zero=False)
    #segments = felzenszwalb(image, scale=200, sigma = 5)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- 10 segments")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img, segments))
    #plt.imshow(segments, cmap='gray')
    plt.axis("off")
    plt.show()
    #segments = np.array(segments, dtype='uint8')
    #segments = cv2.resize(segments, (32, 32), interpolation=cv2.INTER_AREA)
    img_seg.append(segments)

# show the plots
img_seg = np.array(img_seg)
print(img_seg.shape)
#img_seg = img_seg.reshape(-1, 32, 32)
print(img_seg.shape)

query = img_seg[2]

for i in range(len(img_seg)):
    #q_sim = cosine_similarity(img_seg[i], query)
    q_sim = adjusted_rand_score(img_seg[i].flatten(), query.flatten())
    print(q_sim, labels[i])
