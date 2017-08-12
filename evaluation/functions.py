import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')


def upper2Full(a, eps = 0):
    ind = (a<eps)&(a>-eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A


def updateClusters(LLE_node_vals, switch_penalty = 1):
    """
    Uses the Viterbi path dynamic programming algorithm
    to compute the optimal cluster assigments

    Takes in LLE_node_vals matrix and computes the path that minimizes
    the total cost over the path

    Note the LLE's are negative of the true LLE's actually!!!!!
    Note: switch penalty > 0
    """
    (T,num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    ##compute future costs
    for i in xrange(T-2,-1,-1):
        j = i+1
        future_costs = future_cost_vals[j,:]
        lle_vals = LLE_node_vals[j,:]
        for cluster in xrange(num_clusters):
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[cluster] -= switch_penalty
            future_cost_vals[i,cluster] = np.min(total_vals)

    ##compute the best path
    path = np.zeros(T)

    ##the first location
    curr_location = np.argmin(future_cost_vals[0,:] + LLE_node_vals[0,:])
    path[0] = curr_location

    ##compute the path
    for i in xrange(T-1):
        j = i+1
        future_costs = future_cost_vals[j,:]
        lle_vals = LLE_node_vals[j,:]
        total_vals = future_costs + lle_vals + switch_penalty
        total_vals[int(path[i])] -= switch_penalty

        path[i+1] = np.argmin(total_vals)

    ##return the computed path
    return path

def find_matching(confusion_matrix):
    """
    returns the perfect matching
    """
    _,n = confusion_matrix.shape
    path = []
    for i in xrange(n):
        max_val = -1e10
        max_ind = -1
        for j in xrange(n):
            if j in path:
                pass
            else:
                temp = confusion_matrix[i,j]
                if temp > max_val:
                    max_val = temp
                    max_ind = j
        path.append(max_ind)
    return path

def computeF1Score_delete(num_cluster, num_stacked, n, matching_algo, actual_clusters, threshold_algo, save_matrix = False):
    """
    computes the F1 scores and returns a list of values
    """
    F1_score = np.zeros(num_cluster)
    for cluster in xrange(num_cluster):
        matched_cluster = matching_algo[cluster]
        true_matrix = actual_clusters[cluster]
        estimated_matrix = threshold_algo[matched_cluster]
        if save_matrix:
            np.savetxt("estimated_matrix_cluster=" + str(cluster)+".csv",estimated_matrix,delimiter = ",", fmt = "%1.4f")
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in xrange(num_stacked*n):
            for j in xrange(num_stacked*n):
                if estimated_matrix[i,j] == 1 and true_matrix[i,j] != 0:
                    TP += 1.0
                elif estimated_matrix[i,j] == 0 and true_matrix[i,j] == 0:
                    TN += 1.0
                elif estimated_matrix[i,j] == 1 and true_matrix[i,j] == 0:
                    FP += 1.0
                else:
                    FN += 1.0
        precision = (TP)/(TP + FP)
        recall = TP/(TP + FN)
        f1 = (2*precision*recall)/(precision + recall)
        F1_score[cluster] = f1
    return F1_score

def compute_confusion_matrix(num_clusters,clustered_points_algo, sorted_indices_algo):
    """
    computes a confusion matrix and returns it
    """
    seg_len = 200
    true_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for point in xrange(len(clustered_points_algo)):
        cluster = clustered_points_algo[point]


        ##CASE G: ABBACCCA
        # num = (int(sorted_indices_algo[point]/seg_len) )
        # if num in [0,3,7]:
        # 	true_confusion_matrix[0,cluster] += 1
        # elif num in[1,2]:
        # 	true_confusion_matrix[1,cluster] += 1
        # else:
        # 	true_confusion_matrix[2,cluster] += 1

        ##CASE F: ABCBA
        # num = (int(sorted_indices_algo[point]/seg_len))
        # num = min(num, 4-num)
        # true_confusion_matrix[num,cluster] += 1

        #CASE E : ABCABC
        num = (int(sorted_indices_algo[point]/seg_len) %num_clusters)
        true_confusion_matrix[num,cluster] += 1

    ##CASE D : ABABABAB
    # num = (int(sorted_indices_algo[point]/seg_len) %2)
    # true_confusion_matrix[num,cluster] += 1

    ##CASE C:
    # num = (sorted_indices_algo[point]/seg_len)
    # if num < 15:
    # 	true_confusion_matrix[0,cluster] += 1
    # elif num < 20:
    # 	true_confusion_matrix[1,cluster] += 1
    # else:
    # 	true_confusion_matrix[0,cluster] += 1

    ##CASE B :
    # if num > 4:
    # 	num = 9 - num
    # true_confusion_matrix[num,cluster] += 1

    ##CASE A : ABA
    # if sorted_indices_algo[point] < seg_len:
    # 	true_confusion_matrix[0,cluster] += 1

    # elif sorted_indices_algo[point] <3*seg_len:
    # 	true_confusion_matrix[1,cluster] += 1
    # else:
    # 	true_confusion_matrix[0,cluster] += 1

    return true_confusion_matrix

def computeF1_macro(confusion_matrix,matching, num_clusters):
    """
    computes the macro F1 score
    confusion matrix : requres permutation
    matching according to which matrix must be permuted
    """
    ##Permute the matrix columns
    permuted_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for cluster in xrange(num_clusters):
        matched_cluster = matching[cluster]
        permuted_confusion_matrix[:,cluster] = confusion_matrix[:,matched_cluster]
    ##Compute the F1 score for every cluster
    F1_score = 0
    for cluster in xrange(num_clusters):
        TP = permuted_confusion_matrix[cluster,cluster]
        FP = np.sum(permuted_confusion_matrix[:,cluster]) - TP
        FN = np.sum(permuted_confusion_matrix[cluster,:]) - TP
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f1 = stats.hmean([precision,recall])
        F1_score += f1
    F1_score /= num_clusters
    return F1_score

def computeNetworkAccuracy(matching,train_cluster_inverse, num_clusters):
    """
    Takes in the matching for the clusters
    takes the computed clusters
    computes the average F1 score over the network
    """
    threshold = 1e-2
    f1 = 0
    for cluster in xrange(num_clusters):
        true_cluster_cov = np.loadtxt("Inverse Covariance cluster ="+ str(cluster) +".csv", delimiter = ",")
        matched_cluster = matching[cluster]
        matched_cluster_cov = train_cluster_inverse[matched_cluster]
        (nrow,ncol) = true_cluster_cov.shape

        out_true = np.zeros([nrow,ncol])
        for i in xrange(nrow):
            for j in xrange(ncol):
                if np.abs(true_cluster_cov[i,j]) > threshold:
                    out_true[i,j] = 1
        out_matched = np.zeros([nrow,ncol])
        for i in xrange(nrow):
            for j in xrange(ncol):
                if np.abs(matched_cluster_cov[i,j]) > threshold:
                    out_matched[i,j] = 1
        np.savetxt("Network_true_cluster=" +str(cluster) + ".csv",true_cluster_cov, delimiter = ",")
        np.savetxt("Network_matched_cluster=" + str(matched_cluster)+".csv",matched_cluster_cov, delimiter = ",")


        ##compute the confusion matrix
        confusion_matrix = np.zeros([2,2])
        for i in xrange(nrow):
            for j in xrange(ncol):
                confusion_matrix[out_true[i,j],out_matched[i,j]] += 1
        f1 += computeF1_macro(confusion_matrix, [0,1],2)
    return f1/num_clusters

############