import numpy as np
import logging
import os
import time





# trans right to left
def kaixiang_optimize(right, left):
    v_c = left.T - np.tile(left[:, -1], (left.shape[1], 1))
    v_cp = right.T - np.tile(right[:, -1], (right.shape[1], 1))
    v_c = v_c[:-1, :]
    v_cp = v_cp[:-1, :]

    A = np.zeros((4,4))

    for i in range(v_c.shape[0]):
        Q = [[0, -v_c[i, 0], -v_c[i, 1], -v_c[i, 2]],
        [v_c[i, 0], 0, -v_c[i, 2], v_c[i, 1]],
        [v_c[i, 1], v_c[i, 2], 0, -v_c[i, 0]],
        [v_c[i, 2], -v_c[i, 1], v_c[i, 0], 0]]

        W = [[0, -v_cp[i, 0], -v_cp[i, 1], -v_cp[i, 2]],
        [v_cp[i, 0], 0, v_cp[i, 2], -v_cp[i, 1]],
        [v_cp[i, 1], -v_cp[i, 2], 0, v_cp[i, 0]],
        [v_c[i, 2], v_cp[i, 1], -v_cp[i, 0], 0]]

        Q = np.array(Q)
        W = np.array(W)
        A += np.dot((Q-W).T, (Q-W))


    w, v = np.linalg.eig(A)
    # print(w)
    q = v[:, 0]
    # print(q)

    R = [[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
    [2*(q[1]*q[2]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])],
    [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]]

    # theta = np.pi/180*(-52 )
    # R = np.array([[np.cos(theta), 0, np.sin(theta)], 
    #     [0, 1, 0], 
    #     [-np.sin(theta), 0, np.cos(theta)]])

    # R = np.array([[1,0, 0],
    #                 [0,np.cos(theta),-np.sin(theta)],
    #                 [0, np.sin(theta),np.cos(theta)]])

    # R = np.array([[np.cos(theta), -np.sin(theta), 0], 
    # [np.sin(theta), np.cos(theta), 0], 
    # [0, 0, 1]])
    t = np.sum(left - np.dot(R, right), axis=1) / left.shape[1]

    return np.array(R), t.reshape((3, 1)) 


# trans A to B
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def evaluate(ret_R, ret_t, left, right):
    n = left.shape[1]
    new_left = (ret_R@right) + ret_t
    err_matrix = new_left - left
    err = new_left - left
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/n)
    i = 0
    while i < n/28:
        tmp = err_matrix[:, 28*i:28*(i+1)]
        tmp = tmp * tmp
        tmp = np.sum(tmp)
        rse = np.sqrt(tmp/28)
        # print(rse)
        i += 1


    # print('matrix is:')

    # print(ret_R)
    # print(ret_t)
    # if rmse < 1e-5:
    #     print("Everything looks good: ", rmse)
    # else:
    #     print("Hmm something doesn't look right: ", rmse)

    return rmse


R = np.load('R.npy')
t = np.load('t.npy')


for k in range(1, 28):
    left = np.load('left_points_' + str(k) + '.npy')
    right = np.load('right_points_' + str(k) + '.npy')

    valid_left = []
    valid_right = []
    for i in range(left.shape[1]):
        if left[0, i] == 0 and left[1, i] == 0 and left[2, i]== 0:
            continue

        elif right[0, i] == 0 and right[1, i] == 0 and right[2, i]== 0:
            continue

        else:
            valid_left.append([left[0, i], left[1, i], left[2, i]])
            valid_right.append([right[0, i], right[1, i], right[2, i]])

    left = np.array(valid_left).T
    right = np.array(valid_right).T


    rmse = evaluate(R, t, left, right)
    print(k, rmse)
