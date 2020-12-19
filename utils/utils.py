import numpy as np

def generate_data(N1, N2, K, sigma1, sigma2, mean1, mean2, negative_class_indicator=0):
    """
    N1: number of datapoints in class 1.
    N2: number of datapoints in class 2.
    K: number of classes.
    sigma1: var of datapoints for class 1.
    sigma2 = var of datapoints for class 1
    mean*: tuple of dim (1,2) indicating position of the class means
    negative_class_indicator: indicates what value should be taken on 
                            when class is negative
    
    returns matrices T, X, and the color coding
    """
    
    np.random.seed(30)
    cov1 = [[sigma1, 0], [0, sigma1]]
    X1 = np.random.multivariate_normal(mean1, cov1, N1)
    c1 = ['red'] * len(X1)
    
    cov2 = [[sigma2, 0], [0, sigma2]]
    X2 = np.random.multivariate_normal(mean2, cov2, N2)
    c2 = ['blue'] * len(X2)
    
    X = np.concatenate((X1, X2))
    color = np.concatenate((c1, c2))

    T = negative_class_indicator * np.ones([len(X), K])
    for n in range(0, len(X)):
        if (n<len(X1)):
            T[n][0] = 1
        if (n>=N1 and n<len(X1)+len(X2)):
            T[n][1] = 1

    T = T.astype(int)

    return T, X, color