from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import linear_regression_ol as lro
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


class KMeansClusterer:
    def __init__(self, k: int, dataset: str = 'stars') -> None:
        self.k = k
        self.dataset_type = dataset
        self.dataset = self.load_dataset(self.dataset_type)
        self.centroids = self.initialize_centroids()

    def load_dataset(self, ds: str = 'KDD') -> Union[pd.DataFrame, np.ndarray]:
        if ds == 'stars':
            dataset = lro.load_dataset('stars')
        elif ds == 'animals':
            dataset = lro.load_dataset('animals')
            dataset = dataset[dataset['Animal'] != 'Brachiosaurus']
            dataset = dataset[dataset['Animal'] != 'Dipliodocus']
            dataset = dataset[dataset['Animal'] != 'Triceratops']

            dataset.drop(columns=['Animal'], inplace=True)
            dataset.reset_index(inplace=True)
            dataset.drop(columns=['index'], inplace=True)
        elif ds == 'KDD':
            dataset = self.load_kdd_dataset('datasets/NSL-KDD/KDDTest+.txt')
            dataset, self.labels = self.pre_process_dataset(dataset)
        return dataset

    def load_kdd_dataset(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, delimiter=',',
                         header=None)
        columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                    'wrong_fragment', 'urgent', 'hot',  'num_failed_logins', 'logged_in', 'num_compromised',
                    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate', 'attack', 'level'])
        df.columns = columns
        return df

    def pre_process_dataset(self, dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        class_DoS = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
                     'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
        class_Probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']

        class_U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']

        class_R2L = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named',
                     'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient',
                     'warezmaster', 'xlock', 'xsnoop']

        dataset['class'] = dataset['attack']
        dataset['class'].replace(class_DoS, value='DoS', inplace=True)
        dataset['class'].replace(class_Probe, value='Probe', inplace=True)
        dataset['class'].replace(class_U2R, value='Privilege', inplace=True)
        dataset['class'].replace(class_R2L, value='Access', inplace=True)
        labels = {"normal": 0, "DoS": 1, "Probe": 2, "Privilege": 3, "Access": 4}
        dataset['class'].replace(labels, inplace=True)

        protocol_labels = {'tcp': 0, 'udp': 1, 'icmp': 2}
        dataset['protocol_type'].replace(protocol_labels, inplace=True)

        # categorical_columns = ['service', 'flag']
        # self.dataset = pd.get_dummies(self.dataset, columns=categorical_columns)

        le = LabelEncoder()
        le.fit_transform(dataset['service'])
        mapping_service = dict(zip(le.classes_, range(len(le.classes_))))
        dataset['service'].replace(mapping_service,
                                   inplace=True)
        le.fit_transform(dataset['flag'])
        mapping_flag = dict(zip(le.classes_, range(len(le.classes_))))
        dataset['flag'].replace(mapping_flag,
                                inplace=True)
        dataset.drop(columns=['attack'], inplace=True)
        dataset, labels = self.split_dataset(dataset, train_size=0.3)
        return dataset, labels

    def split_dataset(self, dataset: pd.DataFrame, train_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        X = dataset.drop(columns=['class'])
        y = dataset['class']
        X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=12)
        dataset = X_train.reset_index()
        dataset.drop(columns=['index'], inplace=True)
        return dataset, y_train

    def initialize_centroids(self) -> np.ndarray:
        indexes = np.arange(self.dataset.shape[0])
        np.random.seed(12)
        centroids = np.random.choice(indexes, size=self.k, replace=False)
        return centroids

    def calculate_distance(self) -> np.ndarray:
        distances = np.ndarray(shape=(self.dataset.shape[0], self.k))

        for i, centroid in enumerate(self.centroids):
            for idx, row in self.dataset.iterrows():
                distance = np.sqrt(np.sum([(row.iloc[column] - self.dataset.iloc[centroid][column])**2
                                           for column in range(self.dataset.shape[1])]))
                distances[idx][i] = distance
        return distances

    def calc_in_cluster_distance(self, clusters: np.ndarray) -> Tuple[List[list], List[list]]:
        mean_of_features = np.ndarray(shape=(self.k, self.dataset.shape[1]))

        for k in range(self.k):
            for column in range(self.dataset.shape[1]):
                mean_of_features[k, column] = np.mean([self.dataset.iloc[idx, column]
                                                       for idx in range(self.dataset.shape[0])
                                                       if clusters[idx] == k])

        k_distances = []
        idx_k_distances = []

        for k in range(self.k):
            distances = []
            idx_distances = []
            for idx, row in self.dataset.iterrows():
                right_cluster = clusters[idx]
                if right_cluster == k:
                    distance = np.sqrt(np.sum([(row.iloc[column] - mean_of_features[right_cluster][column]) ** 2
                                               for column in range(self.dataset.shape[1])]))
                    distances.append(distance)
                    idx_distances.append(idx)
            k_distances.append(distances)
            idx_k_distances.append(idx_distances)
        return k_distances, idx_k_distances

    def find_new_centroids(self, k_distances: List[list], idx_k_distances: List[list]):
        for k in range(self.k):
            smallest_idx = np.argmin(np.array(k_distances[k]))
            centroid_idx = idx_k_distances[k][smallest_idx]
            self.centroids[k] = centroid_idx

    def plot_data(self, clusters: np.ndarray) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if self.dataset_type == 'stars':
            ax.scatter(self.dataset['log.Te'], self.dataset['log.light'], c=clusters)
            ax.set_xlabel('Log of temperature')
            ax.set_ylabel('Log of light')
            plt.show()
        elif self.dataset_type == 'animals':
            ax.scatter(self.dataset['Body Weight'], self.dataset['Brain Weight'], c=clusters)
            ax.set_xlabel('Body weight')
            ax.set_ylabel('Brain weight')
            plt.show()
        else:
            print('This dataset has more than 3 dimensions and plotting is not supported.')

    def plot_two_solutions(self, clusters: np.ndarray, swapped_clusters: np.ndarray) -> None:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)

        if self.dataset_type == 'stars':
            ax1.scatter(self.dataset['log.Te'], self.dataset['log.light'], c=clusters)
            ax1.set_xlabel('Log of temperature')
            ax1.set_ylabel('Log of light')
        elif self.dataset_type == 'animals':
            ax1.scatter(self.dataset['Body Weight'], self.dataset['Brain Weight'], c=clusters)
            ax1.set_xlabel('Body weight')
            ax1.set_ylabel('Brain weight')
        else:
            print('This dataset has more than 3 dimensions and plotting is not supported.')

        ax2 = fig.add_subplot(122)

        if self.dataset_type == 'stars':
            ax2.scatter(self.dataset['log.Te'], self.dataset['log.light'], c=swapped_clusters)
            ax2.set_xlabel('Log of temperature')
            ax2.set_ylabel('Log of light')
            plt.show()
        elif self.dataset_type == 'animals':
            ax2.scatter(self.dataset['Body Weight'], self.dataset['Brain Weight'], c=swapped_clusters)
            ax2.set_xlabel('Body weight')
            ax2.set_ylabel('Brain weight')
            plt.show()
        else:
            print('This dataset has more than 3 dimensions and plotting is not supported.')

    def perform_swapping(self, b: int, clusters: np.ndarray) -> np.ndarray:
        for _ in range(b):
            idx = np.random.randint(self.dataset.shape[0])
            new_cluster = np.random.randint(self.k)
            clusters[idx] = new_cluster
        return clusters

    def do_iteration(self):
        no_convergence = True
        old_centroids = np.empty_like(self.centroids)
        counter = 1

        while no_convergence:
            print(f'Iteration n. {counter}')
            np.copyto(old_centroids, self.centroids)
            distances = self.calculate_distance()

            clusters = np.argmin(distances, axis=1)

            self.plot_data(clusters)
            self.calc_in_cluster_distance(clusters)

            if np.array_equal(old_centroids, self.centroids):
                no_convergence = False

            counter += 1
        print(f'Convergence after {counter} iterations.')
        # print(f'Accuracy: {accuracy_score(self.labels, clusters.reshape(-1, 1))}, '
        #       f'precision: {precision_score(self.labels, clusters.reshape(-1, 1), average="weighted")}'
        #       f', recall: {recall_score(self.labels, clusters.reshape(-1, 1), average="weighted")}')
        # print(f'Confusion matrix: {confusion_matrix(self.labels, clusters.reshape(-1, 1))}')


if __name__ == '__main__':
    kmeans = KMeansClusterer(k=3, dataset='stars')
    kmeans.do_iteration()

