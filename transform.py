import cv2
from utils import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_canonical_shape(keypoints, res):
    dst = np.array([[0, 20], [10, 20], [20, 20]])
    L = keypoints.shape[0]
    if L == 68:
        l, r = 36, 45
    elif L == 98:
        l, r = 60, 72
    else:
        raise ValueError("")
    src = np.array([keypoints[l], keypoints[l] * 0.5 + keypoints[r] * 0.5, keypoints[r]])
    d, z, tform = procrustes(dst, src)
    keypoints = np.dot(keypoints, tform['rotation']) * tform['scale'] + tform['translation']
    gtbox = get_gtbox(keypoints)
    xmin, ymin, xmax, ymax = gtbox
    keypoints -= [xmin, ymin]
    keypoints *= [res / (xmax - xmin), res / (ymax - ymin)]

    return keypoints


def warp(image, src, dst, res, keypoints=None):
    d, Z, meta = procrustes(dst, src)
    M = np.zeros([2, 3], dtype=np.float32)
    M[:2, :2] = meta['rotation'].T * meta['scale']
    M[:, 2] = meta['translation']
    img = cv2.warpAffine(image, M, (res, res))
    if keypoints is not None:
        keypoints = np.dot(keypoints, meta['rotation']) * meta['scale'] + meta['translation']
    return img, keypoints, meta


def crop_from_box(image, box, res, keypoints=None):
    xmin, ymin, xmax, ymax = box
    src = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    dst = np.array([[0, 0], [0, res - 1], [res - 1, 0], [res - 1, res - 1]])

    return warp(image, src, dst, res, keypoints)


def transform_keypoints(kps, tform, inverse=False):
    if inverse:
        new_kps = np.dot(kps - tform['translation'], np.linalg.inv(tform['rotation'] * tform['scale']))
    else:
        new_kps = np.dot(kps, tform['rotation']) * tform['scale'] + tform['translation']

    return new_kps


def show_preds(image, preds, str_tmp):
    # plt.figure()
    # plt.imshow(image)
    # plt.show()idth
    width, height = image.shape[0], image.shape[1]
    image = cv2.UMat(image).get()
    fig, ax = plt.subplots()
    for pred in preds:
        # plt.scatter(pred[:, 0], pred[:, 1], s=10, marker='.', c='r')
        image = cv2.circle(image, (int(pred[0]), int(pred[1])), 1, (0, 255, 0), 2)
    plt.imshow(image)  # pause a bit so that plots are updated
    plt.axis('off')
    fig.set_size_inches(width / 10.0 / 3.0, height / 10.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(str_tmp)
    # plt.show()


def show_preds_gt(image, preds, i):
    plt.figure()
    plt.imshow(image)
    for pred in preds:
        plt.scatter(pred[:, 0], pred[:, 1], s=100, marker='.', c='g')
    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.axis('off')
    # plt.show()
    plt.savefig('C:/Users/admin/Desktop/300Wresult/gt/gt{}.jpg'.format(i), bbox_inches='tight')
    plt.close('all')


def show_preds_pre(image, preds, i):
    plt.figure()
    plt.imshow(image)
    for pred in preds:
        plt.scatter(pred[:, 0], pred[:, 1], s=100, marker='.', c='r')
    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.axis('off')
    # plt.show()
    plt.savefig('C:/Users/admin/Desktop/300Wresult/pre/pre{}.jpg'.format(i), bbox_inches='tight')
    plt.close('all')


def saveimg(image, i):
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    # plt.show()
    plt.savefig('C:/Users/admin/Desktop/300Wresult/img/img{}.jpg'.format(i), bbox_inches='tight')
    plt.close('all')
