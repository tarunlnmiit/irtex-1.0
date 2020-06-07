
import argparse
import traceback
from tqdm import tqdm
from ResNet20FeatureExtractor import ResNet20FeatureExtractor, extract_feature_resnet
import os
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RESNET20 Extractor Test')
    parser.add_argument('--test_image', help='path to test image. defaults to first image in test folder if empty')
    args = parser.parse_args()

    path = 'toy_data/'
    feature_list = []

    test_image = args.test_image

    if test_image:
        try:
            feature = extract_feature_resnet(test_image)
            feature_list.append([test_image.split('/')[-1], feature, [0]])
        except Exception as e:
            print(traceback.print_exc())
            print('failed to load image, using default test...')
            pass
    else:
        print('No query image provided so one image from toy dataset taken as query')

    for file in tqdm(os.listdir(path)):
        descriptor = extract_feature_resnet(os.path.join(path, file))
        feature_list.append([file, descriptor, [0]])

    query = feature_list[0]

    count = 0

    print('comparison with ', query[0])
    for i in feature_list:
        x1 = i[1]
        x2 = query[1]
        # cosine similarity
        feature_list[count][2] = cosine_similarity(x1.reshape(1,-1), x2.reshape(1,-1))

        print(i[0], '\t', '\t', i[2])
        count += 1
