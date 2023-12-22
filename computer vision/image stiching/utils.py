import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_the_N_information(image, detector, n=200):
    if detector == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
    else:
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

    print('The number of keypoints with {} is {}'.format(detector, len(keypoints)))

    # Convierte los keypoints en un array de NumPy para el valor de respuesta
    responses = np.array([kp.response for kp in keypoints])

    # Ordena los índices de los keypoints basados en la respuesta
    indices = np.argsort(responses)[::-1]

    # Reordena los keypoints usando los índices ordenados
    keypoints_sorted = [keypoints[i] for i in indices][:200]
    # import pdb; pdb.set_trace()
    descriptors_sorted = descriptors[indices][:200]

    img_kp = cv2.drawKeypoints(image, keypoints_sorted, None, color=(255,0,0))
    plt.imshow(img_kp)
    plt.show()
    return keypoints_sorted, descriptors_sorted

def create_distance_matrix(descriptor_1, descriptor_2):
    width = descriptor_1.shape[0]
    distance = np.zeros((width, width))
    for i in range(width):
        one_row_matrix = np.reshape(descriptor_1[i], (1, -1)).repeat(width, 0)
        dist = np.sqrt(np.sum((one_row_matrix - descriptor_2) ** 2, 1))
        distance[i] = dist
    return distance

def NN_and_cross_validation(distance_matrix):
    size = distance_matrix.shape[0]
    threshold = .75
    correspondances = []
    for i in range(size):
        line = distance_matrix[i]
        # Obtein the 1NN and the 2NN
        first_and_second = np.sort(line)[:2]
        # Apply 1NN/2NN threshold
        if (first_and_second[0] / first_and_second[1]) < threshold:
            ind = np.argmin(line)
            # Apply cross validation check
            if np.argmin(distance_matrix[:, ind]) == i:
                # correspondances.append([i, ind])
                match = cv2.DMatch(i, ind, first_and_second[0])
                correspondances.append(match)
    return correspondances

def create_correspondances(descriptor_1, descriptor_2):
    dis = create_distance_matrix(descriptor_1, descriptor_2)
    corr = NN_and_cross_validation(dis)
    return corr

# Now we will estimate the homography matrix.
# To do so, first we are going to normalize the points.
def get_points(matches, kp1, kp2):
    indexes1 = np.array([match.queryIdx for match in matches])
    indexes2 = np.array([match.trainIdx for match in matches])

    points1 = np.array([kp1[ind].pt for ind in indexes1])
    points2 = np.array([kp2[ind].pt for ind in indexes2])
    return points1, points2

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    
    dist = np.linalg.norm(shifted_points, axis=1)
    avg_dist = np.mean(dist)

    scale = np.sqrt(2) / avg_dist

    T = np.array([[scale, 0, -scale * centroid[0]],
                [0, scale, -scale * centroid[1]],
                [0, 0, 1]])
    
    points_normalized = np.dot(T, np.vstack((points.T, np.ones((1, points.shape[0])))))
    return points_normalized.T, T

def check_normalized_distance(points_normalized):
    """
    Check the average distance of the normalized points from the origin.
    
    This function calculates the average distance of a set of 2D points from the origin 
    after normalization. It assumes that the points are in homogeneous coordinates and 
    extracts only the x and y coordinates to compute the distances.

    Parameters:
    points_normalized (np.ndarray): An array of normalized points in homogeneous coordinates.

    Returns:
    float: The average distance of the normalized points from the origin.
    """
    # Extract just the x and y coordinates, ignoring the homogeneous coordinate
    points_normalized = points_normalized[:, :2]
    
    # Calculate the new centroid, which should be at the origin after normalization
    centroid_normalized = np.mean(points_normalized, axis=0)
    
    # Ensure that the centroid is indeed at the origin (allowing a small tolerance for floating-point errors)
    assert np.allclose(centroid_normalized, [0, 0], atol=1e-7), "The centroid is not at the origin."

    # Compute the distances from the origin to each normalized point
    dists_normalized = np.linalg.norm(points_normalized, axis=1)
    
    # Calculate the average distance from the origin
    avg_dist_normalized = np.mean(dists_normalized)
    
    return avg_dist_normalized

def DLT_algorithm(points1, points2):
    np1, T1 = normalize_points(points1)
    np2, T2 = normalize_points(points2)
    number_of_points = np1.shape[0]
    A = np.zeros((2 * number_of_points, 9))
    for i in range(number_of_points):
        x, y = np1[i, 0], np1[i, 1]
        xp, yp = np2[i, 0], np2[i, 1]
        A[ 2 * i] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[ 2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H_norm = np.reshape(h, (3, 3))
    H = np.dot(np.linalg.inv(T2), np.dot(H_norm, T1))
    H = H / H[2, 2]
    return H

def RANSAC(points1, points2, N, sigma=1):
    max_inliners = 0
    best_H = None 
    best_inliners = None
    threshold = np.sqrt(5.99 * (sigma ** 2))
    # N es un para metro que se calcula teniendo en cuenta el minimo de puntos de que se necesita para 
    # estimar la recta (4 para homogrfia) y el numero de outliers que se ve que hay en la imagen (10% en
    # nuestro caso). Con esos datos el cuadro nos da 5 iteraciones para conseguir un 90% de inliners.
    # Como thumbrule se utiliza dos veces ese parametro para setear el numero de iteraciones. 
    for _ in range(2 * N):
        indexes = np.random.choice(points1.shape[0], 4, False)
        sample1 = points1[indexes]
        sample2 = points2[indexes]
        H = DLT_algorithm(sample1, sample2)
        points1_homog = np.append(points1, np.ones((points1.shape[0], 1)), 1)
        estimated_points2_homog = np.dot(H, points1_homog.T).T
        estimated_points2 = estimated_points2_homog[:, :2] / estimated_points2_homog[:, 2][:, np.newaxis]
        distances = np.sqrt(np.sum((estimated_points2 - points2) ** 2, axis=1))
        inliners = distances < threshold
        num_inliners = np.sum(inliners)
        if num_inliners > max_inliners:
            max_inliners = num_inliners
            best_H = H
            best_inliners = inliners
    return best_H, best_inliners

def create_panorama(reference_image, transform_image, H):
    hr, wr = reference_image.shape[:2]
    ht, wt = transform_image.shape[:2]

    corners_t = np.float32([[0, 0], [wt, 0], [0, ht], [wt, ht]]).T
    corners_t = np.concatenate((corners_t, np.ones((1, 4))), axis=0)
    corners_t_homog = H.dot(corners_t)
    corners_t_homog = corners_t_homog / corners_t_homog[-1]

    corners_r = np.float32([[0, 0], [wr, 0], [0, hr], [wr, hr]]).T
    all_corners = np.concatenate((corners_t_homog[:2], corners_r), axis=1)

    xmin = int(np.min(all_corners[0]))
    xmax = int(np.max(all_corners[0]))
    ymin = int(np.min(all_corners[1]))
    ymax = int(np.max(all_corners[1]))

    max_width = int(np.linalg.norm(xmax - xmin))
    max_height = int(np.linalg.norm(ymax - ymin))

    tx = min(0, xmin)
    ty = min(0, ymin)
    T = np.array(([[1], [0], [-tx]], [[0], [1], [-ty]], [[0], [0], [1]])).reshape((3, 3))

    transformed_image = cv2.warpPerspective(transform_image, T.dot(H), (max_width, max_height))
    transformed_image[-ty:-ty+hr, -tx:-tx+wr] = reference_image
    return transformed_image