import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def find_frame(traj_dict, frame_number=0):
    local_frame = frame_number
    for key in sorted(traj_dict):
        if local_frame - traj_dict[key] < 0:
            return local_frame, key
        else:
            local_frame -= traj_dict[key]
    total_num_frames = sum(np.array(list(traj_dict.values())).astype(int))
    raise Exception(
        f"frame {frame_number} should not exceed the total number of frames, {total_num_frames}"
    )


# Helper function for LocalOutlierFactor
def topk(a, k):
    """
    Parameters
    ----------
    a : np.ndarray
        array of dim (N,)
    k : int
        specifies which element to partition upon
    Returns
    -------
    np.ndarray of length k containing indices of input array a
    coresponding to the k smallest values in a.
    """
    return np.argpartition(a, k)[:k]


def outlier_search_lof(embeddings, n_outliers, **kwargs):

    # Handle NANs in embeddings
    embeddings = np.nan_to_num(embeddings, nan=0.0)

    clf = LocalOutlierFactor(**kwargs)

    # Array with 1 if inlier, -1 if outlier
    clf.fit_predict(embeddings)

    # Only sorts 1 element of negative_outlier_factors_, namely the element
    # that is position k in the sorted array. The elements above and below
    # the kth position are partitioned but not sorted. Returns the indices
    # of the elements of left hand side of the parition i.e. the top k.
    outlier_inds = topk(clf.negative_outlier_factor_, k=n_outliers)

    outlier_scores = clf.negative_outlier_factor_[outlier_inds]

    # Only sorts an array of size n_outliers
    sort_inds = np.argsort(outlier_scores)

    # Returns n_outlier best outliers sorted from best to worst
    return outlier_inds[sort_inds], outlier_scores[sort_inds]
