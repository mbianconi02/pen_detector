# functions for Pen Identifier
# Provided during assignment: list_images, get_patch_at_point, sample_points_grid, sample_points_around_pen, 
# remove_points_near_border, make_labels_for_points, extract_patches, extract_multiple_images
# All other function were implemented


import os, glob
import numpy as np
import skimage
import scipy

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def list_images(image_dir, filename_expression='*.jpg'):
    filenames = glob.glob(os.path.join(image_dir, filename_expression))
    filenames = sorted(filenames) # important for cross-platform compatiblity
    print(f'Found {len(filenames)} image files in the directory "{image_dir}"')
    return filenames

def get_image_width(I):
    return I.shape[1]

def get_image_height(I):
    return I.shape[0]

def get_image_channels(I):
    return I.shape[2]

def show_annotation(I, p1, p2):
    plt.figure()
    
    # show the image
    # plot point p1 as a green circle, with markersize 10, and label "tip"
    # plot point p2 as a red circle, with markersize 10, and label "end"
    # plot a line starts at one point and end at another. 
    # Use a suitable color and linewidth for better visualization

    plt.imshow(I)
    plt.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), 'g-', linewidth=5)
    plt.plot(p1[0], p1[1], 'go', markersize=10)
    plt.plot(p2[0], p2[1], 'ro', markersize=10)
    plt.legend(['tip', 'end'])
    
    plt.show()
    
def get_patch_at_point(I, p, WIN_SIZE):
    HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])
    p = p.astype(int)   
    P = I[p[1]-HALF_WIN_SIZE[0]:p[1]+HALF_WIN_SIZE[0],p[0]-HALF_WIN_SIZE[1]:p[0]+HALF_WIN_SIZE[1]]
    
    return P

def patch_to_vec(P, FEAT_SIZE):
    x = skimage.transform.resize(P, FEAT_SIZE).flatten()
    
    return x

def sample_points_grid(I, WIN_SIZE):
    # window centers
    W = get_image_width(I)
    H = get_image_height(I)
    
    HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])
    
    step_size = (WIN_SIZE[0]//2, WIN_SIZE[1]//2)
    min_ys = range(0, H-WIN_SIZE[0]+1, step_size[0])
    min_xs = range(0, W-WIN_SIZE[1]+1, step_size[1])
    center_ys = range(HALF_WIN_SIZE[0], H-HALF_WIN_SIZE[0]+1, step_size[0])
    center_xs = range(HALF_WIN_SIZE[1], W-HALF_WIN_SIZE[1]+1, step_size[1])
    centers = np.array(np.meshgrid(center_xs, center_ys))
    centers = centers.reshape(2,-1).T
    centers = centers.astype(float) 
    
    # add a bit of random offset
    centers += np.random.rand(*centers.shape) * 10 
    
    # discard points close to border where we can't extract patches
    centers = remove_points_near_border(I, centers, WIN_SIZE)
    
    return centers

def sample_points_around_pen(I, p1, p2, WIN_SIZE):
    """
    I: Input image
    p1: Point 1, tip of the pen
    p2: Point 2, end of the pen
    """
    Nu = 100 # uniform samples (will mostly be background, and some non-background)
    Nt = 50 # samples at target locations, i.e. near start, end, and middle of pen
    
    HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])
    
    target_std_dev = np.array(HALF_WIN_SIZE[:2])/3 # variance to add to locations

    upoints = sample_points_grid(I, WIN_SIZE)
    idxs = np.random.choice(upoints.shape[0], Nu)
    upoints = upoints[idxs,:]
    
    
    # sample around target locations (tip and end of the pen)
    tpoints1 = np.random.randn(Nt,2)
    tpoints1 = tpoints1 * target_std_dev + p1

    tpoints2 = np.random.randn(Nt,2)
    tpoints2 = tpoints2 * target_std_dev + p2

    # sample over length pen
    alpha = np.random.rand(Nt)
    tpoints3 = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None])
    tpoints3 = tpoints3 + np.random.randn(Nt,2) * target_std_dev
    
    # merge all points
    points = np.vstack((upoints, tpoints1, tpoints2, tpoints3))
    
    # discard points close to border where we can't extract patches
    points = remove_points_near_border(I, points, WIN_SIZE)
    
    return points

def remove_points_near_border(I, points, WIN_SIZE):
    W = get_image_width(I)
    H = get_image_height(I)
    
    HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])
    
    # discard points that are too close to border
    points = points[points[:,0] > HALF_WIN_SIZE[1],:]
    points = points[points[:,1] > HALF_WIN_SIZE[0],:]
    points = points[points[:,0] < W - HALF_WIN_SIZE[1],:]
    points = points[points[:,1] < H - HALF_WIN_SIZE[0],:]
    
    return points

def make_labels_for_points(I, p1, p2, points, WIN_SIZE):
    """ Determine the class label (as an integer) on point distance to different parts of the pen """
    num_points = points.shape[0]
    
    # for all points ....
    
    # ... determine their distance to tip of the pen
    dist1 = points - p1
    dist1 = np.sqrt(np.sum(dist1 * dist1, axis=1))
    
    # ... determine their distance to end of the pen
    dist2 = points - p2
    dist2 = np.sqrt(np.sum(dist2 * dist2, axis=1))

    # ... determine distance to pen middle
    alpha = np.linspace(0.2, 0.8, 100)
    midpoints = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None]) 
    dist3 = scipy.spatial.distance_matrix(midpoints, points)
    dist3 = np.min(dist3, axis=0)
    
    # the class label of a point will be determined by which distance is smallest
    #    and if that distance is at least below `dist_thresh`, otherwise it is background
    dist_thresh = WIN_SIZE[0] * 2./3.

    # store distance to closest point in each class in columns
    class_dist = np.zeros((num_points, 4))
    class_dist[:,0] = dist_thresh
    class_dist[:,1] = dist1
    class_dist[:,2] = dist2
    class_dist[:,3] = dist3
    
    # the class label is now the column with the lowest number
    labels = np.argmin(class_dist, axis=1)
    
    return labels

def count_classes(labels):
    labels = np.array(labels)
    counts = np.array([np.count_nonzero(labels==0), np.count_nonzero(labels==1), np.count_nonzero(labels==2), np.count_nonzero(labels==3)])
    
    return counts

def class_probs(counts):
    p = counts/sum(counts)
    
    return p

def entropy(p):
    p = p[p>0]
    H = -sum(p*np.log(p))
    
    return H

def extract_patches(I, p1, p2, WIN_SIZE, FEAT_SIZE, strategy=None):
    
    # by default, if no strategy is explicitly defined, use strategy 2
    if strategy == 1:
        points = sample_points_grid(I)
    if strategy == 2 or strategy is None:
        points = sample_points_around_pen(I, p1, p2, WIN_SIZE)
    
    # determine the labels of the points
    labels = make_labels_for_points(I, p1, p2, points, WIN_SIZE)
    
    xs = []
    for p in points:
        P = get_patch_at_point(I, p, WIN_SIZE)
        x = patch_to_vec(P, FEAT_SIZE)
        xs.append(x)
    X = np.array(xs)

    return X, labels, points

def extract_multiple_images(idxs, Is, annots, WIN_SIZE, FEAT_SIZE, strategy=None):
    """
    idxs: index
    Is: the list contains all images
    """
    Xs = []
    ys = []
    points = []
    imgids = []

    for step, idx in enumerate(idxs):
        I = Is[idx]
        I_X, I_y, I_points = extract_patches(I, annots[idx,:2], annots[idx,2:], WIN_SIZE, FEAT_SIZE, strategy=strategy)

        classcounts = count_classes(I_y)
        #print(f'image {idx}, class count = {classcounts}')

        Xs.append(I_X)
        ys.append(I_y)
        points.append(I_points)
        imgids.append(np.ones(len(I_y),dtype=int)*idx)

    Xs = np.vstack(Xs)
    ys = np.hstack(ys)
    points = np.vstack(points)
    imgids = np.hstack(imgids)
    
    return Xs, ys, points, imgids





