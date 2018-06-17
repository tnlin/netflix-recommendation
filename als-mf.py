import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def read_movie_titles():
    with open('Project2-data/movie_titles.txt', 'r', encoding = "ISO-8859-1") as fp:
        movie_titles = fp.read().splitlines()

    rule = "(\d+),(.*),(.*)"
    l = []
    for title in movie_titles:
        hit = re.match(rule, title)
        l.append([hit[1], hit[2], hit[3]])
    return pd.DataFrame(l, columns=['id', 'mid', 'title'])

def predict(ratings, similarity,):
    # 注意分母，只有看过该电影的人，才能将其打分加入做CF
    return similarity.dot(ratings) / similarity.dot((ratings>0) +  np.finfo(float).eps)

def rmse(mat_pred, mat_test):
    y_pred = mat_pred[mat_test.nonzero()].flatten()
    y_test = mat_test[mat_test.nonzero()].flatten()
    return np.sqrt(mean_squared_error(y_pred, y_test))

df_title = read_movie_titles()
df_user = pd.read_csv('Project2-data/users.txt', sep=' ', header=None, names=['uid'])
df_train = pd.read_csv('Project2-data/netflix_train.txt', sep=' ', header=None, names=['uid', 'mid', 'rating', 'time'])
df_test = pd.read_csv('Project2-data/netflix_test.txt', sep=' ', header=None, names=['uid', 'mid', 'rating', 'time'])


fake_id = 0
impute = [{'uid': fake_id, 'mid': i, 'rating': 0, 'time': None} for i in range(1, 10000)]
df_test = df_test.append(impute)

rating_train = df_train.pivot(index = 'uid', columns ='mid', values = 'rating').fillna(0)
rating_test = df_test.pivot(index = 'uid', columns ='mid', values = 'rating').fillna(0)
rating_test = rating_test.drop(fake_id)

print("Train shape:", rating_train.shape, "Test shape:", rating_test.shape)

mat_train = np.array(rating_train)
mat_test = np.array(rating_test)

import numpy as np
import time
from numpy import linalg as LA


class MF():
    def __init__(self, X, k, lamb, alpha, iterations):
        """
        Perform matrix factorization to predict empty entries in a matrix.
        
        Arguments
        - X (ndarray)   : user-item rating matrix
        - k (int)       : number of latent dimensions
        - lamb (float)  : regularization parameter
        - alpha (float) : learning rate
        """
        self.X = X
        self.A = (X>0)*1
        self.n_users, self.n_items = X.shape
        self.k = k
        self.alpha = alpha
        self.lamb = lamb
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale=1./self.k, size=(self.n_users, self.k)) # 10000, k
        self.V = np.random.normal(scale=1./self.k, size=(self.n_items, self.k)) # 10000, k
        
        RMSEs = []
        for i in range(self.iterations):
            start_time = time.time()
            self.gradient_descent()
            RMSE = self.rmse()
            RMSEs.append((i, RMSE))

            if i%10==0:
                J = self.J()
                elasping = time.time() - start_time
                print("Iteration: %d; J=%.3f; RMSE = %.4f; %.2f seconds" % (i+1, J, RMSE, elasping))

        return RMSEs
    
    def J(self):
        loss = self.predict() - self.X
        J = 0.5 * LA.norm(np.multiply(self.A, loss)) + self.lamb * LA.norm(self.U) + self.lamb * LA.norm(self.V)
        return J
    
    def rmse(self):
        """
        A function to compute the total root mean square error
        """
        mat_pred = self.predict()
        y_pred = mat_pred[R.nonzero()].flatten()
        y_test = R[R.nonzero()].flatten()
        return np.sqrt(mean_squared_error(y_pred, y_test))

    def gradient_descent(self):
        """
        Perform graident descent
        """
        loss = self.predict() - self.X
        dU = np.dot(np.multiply(self.A, loss), self.V) + 2 * self.lamb * self.U
        dV = np.dot(np.multiply(self.A, loss).T, self.U) + 2 * self.lamb * self.V
        
        self.U -= self.alpha * dU
        self.V -= self.alpha * dV

    def predict(self):
        return self.U.dot(self.V.T)


R = mat_train
k = 20
lamb = 0.1
mf = MF(R, k=k, lamb=lamb, alpha=0.0001, iterations=150)
rmses = mf.train()

mat_pred = mf.predict()
print('User-based CF: RMSE: ' + str(rmse(mat_pred, mat_test)))

df = pd.DataFrame(rmses, columns=['index', 'RMSE'])['RMSE']
RMSE = rmses[-1][1]
df.plot(figsize=(8,4), legend=True, title=("RMSE=%.3f (k=%d, lambda=%.2f)" % (RMSE, k, lamb)))