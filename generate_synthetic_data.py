import numpy as np
from snap import *
import os

def genInvCov(size, low=0.3, upper=0.6, portion=0.2, symmetric=True):
    portion = portion / 2
    S = np.zeros((size, size))

    G = GenRndGnm(PNGraph, size, int((size * (size - 1)) * portion))
    for EI in G.Edges():
        value = 2*(np.random.randint(2)-0.5)*(low + (upper - low) * np.random.rand(1)[0])
        # print value
        S[EI.GetSrcNId(), EI.GetDstNId()] = value

    if symmetric:
        S = S + S.T

    # vals = alg.eigvalsh(S)
    # S = S + (0.1 - vals[0])*np.identity(size)

    return np.matrix(S)


def genRandInv(size, low=0.3, upper=0.6, portion=0.2):
    S = np.zeros((size, size))
    for i in xrange(size):
        for j in xrange(size):
            if np.random.rand() < portion:
                value = (np.random.randint(2) - 0.5) * 2 * (low + (upper - low) * np.random.rand(1)[0])
                S[i, j] = value
    return np.matrix(S)


def generate_inverse(rand_seed):

    np.random.seed(rand_seed)

    ##Generate all the blocks
    for block in xrange(segment_size):
        if block ==0:
            block_matrices[block] = genInvCov(size = num_signals, portion = sparsity_inv_matrix, symmetric = (block == 0))
        else:
            block_matrices[block] = genRandInv(size = num_signals, portion = sparsity_inv_matrix)

    ##Initialize the inverse matrix
    inv_matrix = np.zeros([segment_size * num_signals, segment_size * num_signals])

    ##go through all the blocks
    for block_i in xrange(segment_size):
        for block_j in xrange(segment_size):
            block_num = np.abs(block_i - block_j)
            if block_i > block_j:
                inv_matrix[block_i * num_signals:(block_i + 1) * num_signals, block_j * num_signals:(block_j + 1) * num_signals] = block_matrices[block_num]
            else:
                inv_matrix[block_i * num_signals:(block_i + 1) * num_signals, block_j * num_signals:(block_j + 1) * num_signals] = np.transpose(block_matrices[block_num])

    ##print out all the eigenvalues
    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)

    ##Make the matrix positive definite
    inv_matrix = inv_matrix + (0.1 + abs(lambda_min))*np.identity(num_signals * segment_size)

    eigs, _ = np.linalg.eig(inv_matrix)
    print "Modified Eigenvalues are:", np.sort(eigs)

    return inv_matrix


##Notes
##Generate a random Erdos-Randyii matrix
##Check if the matrix satisfies our requirements
##If does not satisfy then add a lambda*I term to make SPD
##check for the conditioning number - and fix it if necessary
##How do we enforce the requirement the covariance matrices are different??

if __name__ == "__main__":

    ##Parameters to play with

    window_size = 5
    number_of_sensors = 5
    sparsity_inv_matrix = 0.2
    rand_seed = 10
    number_of_clusters = 3
    cluster_ids = [0,1,0]
    break_points = np.array([1,2,3])*200
    save_inverse_covarainces = True
    os.mkdir('data/')
    out_file_name = "data/Synthetic Data Matrix rand_seed =[0,1] generated2.csv"
    ###########################################################

    block_matrices = {} ##Stores all the block matrices
    segment_size = window_size
    num_signals = number_of_sensors
    sparsity_inv_matrix = sparsity_inv_matrix
    block_matrices = {} ##Stores all the block matrices
    seg_ids = cluster_ids


    ############GENERATE POINTS
    num_clusters = number_of_clusters
    cluster_mean = np.zeros([num_signals, 1])
    cluster_mean_stacked = np.zeros([num_signals * segment_size, 1])

    ##Generate two inverse matrices
    cluster_inverses = {}
    cluster_covariances = {}
    for cluster in xrange(num_clusters):
        cluster_inverses[cluster] = generate_inverse(rand_seed = cluster)
        cluster_covariances[cluster] = np.linalg.inv(cluster_inverses[cluster])
        if save_inverse_covarainces:
            np.savetxt("data/Inverse Covariance cluster ="+ str(cluster) +".csv", cluster_inverses[cluster],delimiter= ",",fmt='%1.6f')
            np.savetxt("data/Covariance cluster ="+ str(cluster) +".csv", cluster_covariances[cluster],delimiter= ",",fmt='%1.6f')

    ##Data matrix
    Data = np.zeros([break_points[-1], num_signals])
    Data_stacked = np.zeros([break_points[-1] - segment_size + 1, num_signals * segment_size])

    cluster_point_list = []
    for counter in xrange(len(break_points)):
        break_pt = break_points[counter]
        cluster = seg_ids[counter]
        if counter == 0:
            old_break_pt = 0
        else:
            old_break_pt = break_points[counter-1]
        for num in xrange(old_break_pt,break_pt):
            ##generate the point from this cluster
            if num == 0:
                cov_matrix = cluster_covariances[cluster][0:num_signals, 0:num_signals]##the actual covariance matrix
                new_mean = cluster_mean_stacked[num_signals * (segment_size - 1):num_signals * segment_size]
                ##Generate data
                print "new mean is:", new_mean
                print "size_blocks:", num_signals
                print "cov_matrix is:", cov_matrix
                new_row = np.random.multivariate_normal(new_mean.reshape(num_signals), cov_matrix)
                Data[num,:] = new_row
            elif num < segment_size:
                ##The first section
                cov_matrix = cluster_covariances[cluster][0:(num+1) * num_signals, 0:(num + 1) * num_signals] ##the actual covariance matrix
                n = num_signals
                Sig22 = cov_matrix[(num)*n:(num+1)*n,(num)*n:(num+1)*n]
                Sig11 = cov_matrix[0:(num)*n,0:(num)*n]
                Sig21 = cov_matrix[(num)*n:(num+1)*n,0:(num)*n]
                Sig12 = np.transpose(Sig21)
                cov_mat_tom = Sig22 - np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),Sig12) #sigma2|1
                log_det_cov_tom = np.log(np.linalg.det(cov_mat_tom))# log(det(sigma2|1))
                inv_cov_mat_tom = np.linalg.inv(cov_mat_tom)# The inverse of sigma2|1

                ##Generate data
                a = np.zeros([(num) * num_signals, 1])
                for idx in xrange(num):
                    a[idx * num_signals:(idx + 1) * num_signals, 0] = Data[idx, :].reshape([num_signals])
                new_mean = cluster_mean + np.dot(np.dot(Sig21,np.linalg.inv(Sig11)), (a - cluster_mean_stacked[0:(num) * num_signals, :]))
                new_row = np.random.multivariate_normal(new_mean.reshape(num_signals), cov_mat_tom)
                Data[num,:] = new_row
            else:
                cov_matrix = cluster_covariances[cluster]##the actual covariance matrix
                n = num_signals
                Sig22 = cov_matrix[(segment_size - 1) * n:(segment_size) * n, (segment_size - 1) * n:(segment_size) * n]
                Sig11 = cov_matrix[0:(segment_size - 1) * n, 0:(segment_size - 1) * n]
                Sig21 = cov_matrix[(segment_size - 1) * n:(segment_size) * n, 0:(segment_size - 1) * n]
                Sig12 = np.transpose(Sig21)
                cov_mat_tom = Sig22 - np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),Sig12) #sigma2|1
                log_det_cov_tom = np.log(np.linalg.det(cov_mat_tom))# log(det(sigma2|1))
                inv_cov_mat_tom = np.linalg.inv(cov_mat_tom)# The inverse of sigma2|1

                ##Generate data
                a = np.zeros([(segment_size - 1) * num_signals, 1])
                for idx in xrange(segment_size-1):
                    a[idx * num_signals:(idx + 1) * num_signals, 0] = Data[num - segment_size + 1 + idx, :].reshape([num_signals])

                new_mean = cluster_mean + np.dot(np.dot(Sig21,np.linalg.inv(Sig11)), (a - cluster_mean_stacked[0:(segment_size - 1) * num_signals, :]))
                new_row = np.random.multivariate_normal(new_mean.reshape(num_signals), cov_mat_tom)
                Data[num,:] = new_row

    print "done with generating the data!!!"
    print "length of generated Data is:", Data.shape[0]

    ##save the generated matrix
    np.savetxt(out_file_name, Data, delimiter=",", fmt='%1.4f')