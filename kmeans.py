import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bandb

if __name__=="__main__":

    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', type=int, default=20, help="Number of samples.")
    parser.add_argument('--k', dest='k', type=int, default=3, help="Number of clusters.")
    args = parser.parse_args()
    
    """
    X: n x m
       n: number of samples
       m: number of features
    """
    np.random.seed(715)
    n = args.n  # Number of samples
    k = args.k  # Number of clusters
    X = np.random.random([n, 2])

    
    def scatterwlabels(X, ylab, center, namestr=""):
        """
        Scatter Plot with label colors
        """
        colors = cm.rainbow(np.linspace(0, 1, ci+1))
        plt.figure()
        for i, coord in enumerate(X):
            x = X[i, 0]
            y = X[i, 1]
            if center[i] == 1:
                plt.scatter(x, y, marker='x', color=colors[ylab[i].astype(int)])
            else:
                plt.scatter(x, y, marker='o', color=colors[ylab[i].astype(int)])
            plt.text(x+0.015, y+0.015, str(i), fontsize=9)
        plt.savefig("kmeans"+namestr+"-"+str(n)+".png")
    
    
    def distance(xi, xj):
        return np.linalg.norm(xi - xj, 2)
    
    
    # Formulate k-means as B&B
    obj = ['min']
    
    for i in range(n):
        for j in range(n):
            obj += [distance(X[i, :], X[j, :])]
        
    for i in range(n):
        obj += [0] * n
        
    fixed = [-1] * (n * n + n)  # all variables are not fixed to either 0 or 1
    
    number = n * n + n
    
    def oneij(ij, n):
        yvec = [0]*(ij) + [1] + [0]*(n*n-ij-1)
        xvec = [0]*j + [-1] + [0]*(n-j-1)
        return yvec + xvec
    
    def yij(ij, n):
        return [0]*(ij) + [1] + [0]*(n*n-ij-1) + [0]*n
    
    def xivec(i, n):
        return [0]*n*n + [0]*i + [1] + [0]*(n-i-1)
    
    cons = []
    
    """ \sum_{i=1}^n x_i = k """
    cons += [[0]*n*n + [1]*n + ["="] + [k]]
    
    """ \sum_{j=1}^n y_{ij} = 1, \forall i \in [n] """
    for i in range(n):
        cons += [[0]*i*n + [1]*n + [0]*(n-i-1)*n + [0]*n + ["="] + [1]]
        
    """  y_{ij} \leq x_j, \forall i,j \in [n] """
    for i in range(n):
        for j in range(n):
            cons += [oneij(i*n+j, n) + ["<="] + [0]]
    
    """ x_i \in {0, 1}, \forall i \in [n] """
    for i in range(n):
        cons += [xivec(i, n) + ["<="] + [1]]
        cons += [xivec(i, n) + [">="] + [0]]
    
    """ x_{ij} \in {0, 1}, \forall i,j \in [n] """
    for i in range(n):
        for j in range(n):
            cons += [yij(i*n+j, n) + ["<="] + [1]]
            cons += [yij(i*n+j, n) + [">="] + [0]]
            
    s = bandb.BandB(number, cons, obj)
    solution, opt = s.bandb()
    
    # Convert solution to labels
    ymat = np.array(solution[:n*n]).reshape([n, n])
    ylab = np.zeros([n, ])
    xvec = solution[-n:]
    ci = 0
    for ii, ki in enumerate(xvec):
        if ki == 1:
            for i in range(n):
                if ymat[i, ii] == 1:
                    ylab[i] = ci
            ci += 1
    
    print(ylab.T.astype(int))
    
    scatterwlabels(X, ylab, solution[n*n:])
    
    # Tree Size


#%%
    
    from sklearn.cluster import KMeans
    import numpy as np
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    ylab = kmeans.labels_
    print(ylab)
    scatterwlabels(X, ylab, center=np.zeros([len(ylab),]), namestr="-pytest")
    #kmeans.predict([[0, 0], [4, 4]])
    #kmeans.cluster_centers_