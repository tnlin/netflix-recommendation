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

start_time = time.time()
print("Calculating cosine similarity")
mat_train = np.array(rating_train)
mat_test = np.array(rating_test)
mat_sim = cosine_similarity(mat_train, dense_output=False)
print("--- %s seconds ---" % (round(time.time() - start_time, 2)))

print("User-based CF: Preditioning...")
mat_pred = predict(mat_train, mat_sim)

print('User-based CF: RMSE: ' + str(rmse(mat_pred, mat_test)))
print("--- %s seconds ---" % (round(time.time() - start_time, 2)))

## See what's really happening
# prediction = mat_pred[mat_test.nonzero()].flatten()
# ground_truth = mat_test[mat_test.nonzero()].flatten()