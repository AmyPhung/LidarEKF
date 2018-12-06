import numpy as np
import random


def pol2cart(data):
    return np.vstack((np.cos(data[0]), np.sin(data[0]))) * data[1]


def process(data):
    '''Remove all zero ranges and convert to cartesian'''
    carts = pol2cart(data[:, data[1] != 0])
    return carts[:, carts[0].argsort()]


def circle3(data):
    '''Find the center of a circle from 3 points'''
    deltas = (data[:, 1:] - data[:, :-1])
    slopes = deltas[1] / deltas[0]
    cx = (slopes.prod() * (data[1, 0] - data[1, 2]) + slopes[1] *
          (data[0, 0] + data[0, 1]) - slopes[0] * (data[0, 1] + data[0, 2]))\
        / 2 / (slopes[1] - slopes[0])
    cy = ((data[0, 0] + data[0, 1]) / 2 - cx) / \
        slopes[0] + (data[1, 0] + data[1, 1]) / 2
    center = np.array([cx, cy])
    radius = np.sqrt(((data[:, 0] - center) ** 2).sum())
    return center, radius


def rcirc(data, r, k, d, minPts):
    '''RANSAC data for a circle of radius r.'''
    if k < 0 or data.shape[1] < minPts:
        return np.array([]), data

    pt1 = data[:, np.random.randint(data.shape[1])]

    candidates = data[:, np.linalg.norm(
        data - pt1[:, np.newaxis], axis=0) < 2 * (r + d)]

    if candidates.shape[1] < minPts:
        return rcirc(data, r, k - 1, d, minPts)

    c, r1 = circle3(candidates[:, sorted(
        random.sample(range(candidates.shape[1]), 3))])

    if np.abs(r1 - r) > .05:
        return rcirc(data, r, k - 1, d, minPts)

    mask = np.abs(np.linalg.norm((data - c[:, np.newaxis]), axis=0) - r1) < d

    if mask.sum() < minPts:
        return rcirc(data, r, k - 1, d, minPts)

    return c, data[:, ~mask]


def ransac(data, k, d, gap, minPts, bestInliers=np.reshape([], (2, 0)), bestOutliers=np.reshape([], (2, 0))):
    '''Classify points into either on the best line segment or not.'''

    if k == 0:
        if bestInliers.size == 0:
            return bestInliers, data
        else:
            return bestInliers, bestOutliers
    if data.shape[1] < 2:
        return bestInliers, data

    samp = data[:, sorted(random.sample(range(data.shape[1]), 2))]
    kLine = samp[:, 1] - samp[:, 0]

    data = data[:, np.dot(kLine[np.newaxis, :], data).ravel().argsort()]
    kLineNorm = kLine / np.linalg.norm(kLine)

    normVector = np.reshape([-kLineNorm[1], kLineNorm[0]], (1, 2))
    err = np.dot(normVector, data - samp[:, 0:1])

    mask = np.abs(err.ravel()) <= d
    inliers = data[:, mask]
    outliers = data[:, ~mask]

    if inliers.shape[1] > 2:
        dists = np.dot(kLineNorm, inliers - data[:, 0:1])
        gapIdx = ((dists[1:] - dists[:-1] > gap).nonzero())[0]
        sections = np.stack(
            (np.append(0, gapIdx + 1), np.append(gapIdx + 1, inliers.shape[1])))
        section = sections[:, (sections[1] - sections[0]).argmax()]
        outliers = np.hstack(
            (outliers, inliers[:, :section[0]], inliers[:, section[1]:]))
        inliers = inliers[:, section[0]:section[1]]

    if inliers.shape[1] > bestInliers.shape[1] and inliers.shape[1] >= minPts:
        return ransac(data, k - 1, d, gap, minPts, inliers, outliers)

    else:
        return ransac(data, k - 1, d, gap, minPts, bestInliers, bestOutliers)


def multipleRansac(data, k, d, gap, minPts, groups=[]):
    '''Find many groups of points classified by line segment.'''

    inliers, outliers = ransac(data, k, d, gap, minPts)

    if inliers.size == 0:
        return np.array(groups), outliers

    groups.append(inliers[:, [0, -1]])
    return multipleRansac(outliers, k, d, gap, minPts, groups)
