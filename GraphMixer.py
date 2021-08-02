import numpy as np
import copy
import scipy.sparse as sp

def assign_random_weights(A):
    X = np.random.random(size=(A.shape[0],A.shape[0]))
    W = np.multiply(X, A)
    return (W + W.T)/2

def turn_to_directed(mat, directed = 0.0, weighted = 0):

    if not isinstance(mat, np.ndarray):
        raise Exception('Wrong input parsed to turn_to_directed function!')

    A = copy.deepcopy(mat)
    if directed == 0.0:
        if not weighted:
            a = A.astype(bool)
        else:
            a = A.astype(float)
        return sp.csr_matrix(a)

    np.fill_diagonal(A, 0)
    rows, cols = A.nonzero()
    edgeset = set(zip(rows, cols))
    upper = np.array([l for l in edgeset if l[0]<l[1]])
    dircount = 0

    random_tosses = np.random.random(len(upper))
    condition1 = (random_tosses >= directed/2.0) & (random_tosses < directed)
    condition2 = (random_tosses <= directed/2.0) & (random_tosses < directed)
    indices_where_upper_is_removed = np.where(condition1 == True)[0]
    indices_where_lower_is_removed = np.where(condition2 == True)[0]


    u_xdata = [u[0] for u in upper[indices_where_upper_is_removed]]
    u_ydata = [u[1] for u in upper[indices_where_upper_is_removed]]
    A[u_xdata, u_ydata] = 0

    l_xdata = [u[1] for u in upper[indices_where_lower_is_removed]]
    l_ydata = [u[0] for u in upper[indices_where_lower_is_removed]]
    A[l_xdata, l_ydata] = 0

    a = sp.csr_matrix(A)
    #get_symmetry_index(a)
    return a

def get_symmetry_index(a):
    a = a.astype(bool)
    symmetrized = a + a.T

    difference = symmetrized.astype(int) - a.astype(int)
    difference.eliminate_zeros()
    symm_index = 1 - difference.nnz/symmetrized.nnz*2
    # symm_index is 1 for a symmetrix matrix and 0 for an asymmetric one
    return symm_index

def symmetric_component(A, is_weighted):
    a = A.astype(bool).A
    symm_mask = np.bitwise_and(a, a.T)
    if not is_weighted:
        return symm_mask

    return np.multiply(symm_mask, A.A)

def non_symmetric_component(A, is_weighted):
    return A.astype(float) - symmetric_component(A, is_weighted).astype(float)


def adj_random_rewiring_iom_preserving(a, is_weighted, r=10):
    s = symmetric_component(a, is_weighted)
    #plt.matshow(s)
    rs = turn_to_directed(s, directed = 1.0, weighted = is_weighted)
    #plt.matshow(s)
    #plt.matshow(rs.A)
    rows, cols = rs.A.nonzero()
    edgeset = set(zip(rows, cols))
    upper = [l for l in edgeset]
    source_nodes = [e[0] for e in upper]
    target_nodes = [e[1] for e in upper]

    double_edges = len(upper)

    i=0

    while i < double_edges*r:
        good_choice = 0
        while not good_choice:
            ind1, ind2 = np.random.choice(double_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len(set([n1,n2,n3,n4])) == 4:
                good_choice = 1

        w1 = s[n1,n2]
        w2 = s[n2,n1]
        w3 = s[n3,n4]
        w4 = s[n4,n3]

        if s[n1,n3] + s[n1,n4] + s[n2,n3] + s[n2,n4] == 0:

            s[n1,n4] = w1
            s[n4,n1] = w2
            s[n2,n3] = w3
            s[n3,n2] = w4

            s[n1,n2] = 0
            s[n2,n1] = 0
            s[n3,n4] = 0
            s[n4,n3] = 0

            target_nodes[ind1], target_nodes[ind2] = n4, n2
            i += 1

    #plt.matshow(s)
    #print ('Rewiring single connections...')

    ns = non_symmetric_component(a, is_weighted)

    #plt.matshow(ns)
    rows, cols = ns.nonzero()
    edges = list((set(zip(rows, cols))))
    source_nodes = [e[0] for e in edges]
    target_nodes = [e[1] for e in edges]
    single_edges = len(edges)

    i=0

    while i < single_edges*r:
        good_choice = 0
        while not good_choice:
            ind1, ind2 = np.random.choice(single_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len(set([n1,n2,n3,n4])) == 4:
                good_choice = 1

        w1 = ns[n1,n2]
        w2 = ns[n3,n4]

        checklist = [ns[n1,n3], ns[n1,n4], ns[n2,n3], ns[n2,n4],
                     ns[n3,n1], ns[n4,n1], ns[n3,n2], ns[n4,n2],
                     s[n3,n1], s[n4,n1], s[n3,n2], s[n4,n2]]

        if checklist.count(0) == 12:

            ns[n1,n4] = w1
            ns[n3,n2] = w2

            ns[n1,n2] = 0
            ns[n3,n4] = 0

            i += 1

            target_nodes[ind1], target_nodes[ind2] = n4, n2
            #print(get_symmetry_index(sp.csr_matrix(A)))

    res = s + ns
    if not is_weighted:
        res = res.astype(bool)

    return sp.csr_matrix(res)
