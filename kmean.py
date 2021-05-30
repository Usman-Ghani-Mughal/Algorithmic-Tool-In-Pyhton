import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import mean

class Kmeans:
    def __init__(self):
        self.k = 0
        self.centroids = []
        self.centroids_groups = []
        self.flag = True
        self.data = []
        self.data_1=[]

    def fit(self, dataframe, k):
        total_records = len(dataframe.index)
        self.k = k
        for i in range(k):
            self.centroids_groups.append([])

        for i in range(k):
            random_point = int(np.random.randint(low=0, high=total_records, size=1))
            self.centroids.append(random_point)

        for row_index in range(0, total_records):
            some_data_point = dataframe.iloc[row_index, :].tolist()

            for centroid_index in self.centroids:
                centroid = df.iloc[centroid_index, :].tolist()
                distance_1 = self.find_distance(centroid, some_data_point)
                centroid = df.iloc[self.centroids[1], :].tolist()
                distance_2 = self.find_distance(centroid, some_data_point)

                if distance_1 < distance_2:
                    self.centroids_groups[0].append(row_index)
                else:
                    self.centroids_groups[1].append(row_index)
                break
        x = []
        y = []
        for index in self.centroids_groups[0]:
            centroid = df.iloc[index, :].tolist()
            x.append(centroid[0])
            y.append(centroid[1])
        plt.scatter(x, y, c='red')
        X = []
        Y = []
        for index in self.centroids_groups[1]:
            centroid = df.iloc[index, :].tolist()
            X.append(centroid[0])
            Y.append(centroid[1])
        plt.scatter(X, Y, c='green')

        plt.show()
        for x_,y_ in zip(x,y):
            self.data.append([x_,y_])
        for x_,y_ in zip(X,Y):
            self.data.append(x_,y_)


    def find_mean(self, d_list):
        return mean(d_list)





        print(self.centroids_groups[0])
        print('************************')
        print(self.centroids_groups[1])



    def find_distance(self, centroid, point):
        try:
            distance = math.sqrt(math.pow(centroid[0] - point[0], 2) + math.pow(centroid[1] - point[1], 2))
            return distance
        except:
            print("Some Error")





# df = pd.read_csv('clustering_data_sets\\train.csv')
# obj = Kmeans()
# obj.fit(df, 2)


