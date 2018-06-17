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

def preprocess(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df.time.dt.year
    df['month'] = df.time.dt.month
    df['day'] = df.time.dt.day
    df['dayofweek'] = df.time.dt.dayofweek
    df['dayofyear'] = df.time.dt.dayofyear
    df['weekofyear'] = df.time.dt.weekofyear
    return df.drop(['rating', 'time', 'mid'], axis=1)


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

# Building date statistic feature for user-similairy matrix
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all = df_all.dropna()
df_all = preprocess(df_all)
df_onehot = pd.get_dummies(df_all, columns=['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear'])
df_stats = df_onehot.groupby('uid').mean()

print("Calculating cosine similarity")
start_time = time.time()
mat_train = np.array(rating_train.join(df_stats))
mat_test = np.array(rating_test.join(df_stats))
mat_sim = cosine_similarity(mat_train, dense_output=False)
print("--- %s seconds ---" % (round(time.time() - start_time, 2)))

print("User-based CF: Preditioning...")
mat_pred = predict(mat_train, mat_sim)

print('User-based CF: RMSE: ' + str(rmse(mat_pred, mat_test)))
print("--- %s seconds ---" % (round(time.time() - start_time, 2)))

## See what's really happening
# prediction = mat_pred[mat_test.nonzero()].flatten()
# ground_truth = mat_test[mat_test.nonzero()].flatten()