import numpy as np
from kmeans import KMeansClusterer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class SimulatedAnnealer:
    def __init__(self, t0: float, tf: float, r: float, b: int, k: int) -> None:
        self.t0 = t0
        self.tf = tf
        self.r = r
        self.b = b
        self.kmeans = KMeansClusterer(k=k, dataset='KDD')

    def choose_candidate_solution(self, f0: float, f1: float, ti: float) -> int:
        if f1 < f0:
            return 1
        else:
            if np.exp((-(f1 - f0))/ti) > np.random.uniform(0, 1):
                return 1
            else:
                return 0

    def run_iteration(self) -> None:
        counter = 1
        candidate_solution = 0

        while self.t0 > self.tf:
            print(f'Iteration n. {counter}')
            distances = self.kmeans.calculate_distance()
            clusters = np.argmin(distances, axis=1)

            swapped_clusters = np.empty_like(clusters)

            np.copyto(swapped_clusters, clusters)
            swapped_clusters = self.kmeans.perform_swapping(self.b, swapped_clusters)

            self.kmeans.plot_two_solutions(clusters, swapped_clusters)
            distances, idx_distances = self.kmeans.calc_in_cluster_distance(clusters)
            sw_distances, sw_idx_distances = self.kmeans.calc_in_cluster_distance(swapped_clusters)

            distance = np.sum([np.sum(np.array(distances[_])) for _ in range(self.kmeans.k)])
            sw_distance = np.sum([np.sum(np.array(sw_distances[_])) for _ in range(self.kmeans.k)])

            candidate_solution = self.choose_candidate_solution(distance, sw_distance, self.t0)

            if candidate_solution == 1:
                self.kmeans.find_new_centroids(sw_distances, sw_idx_distances)
            elif candidate_solution == 0:
                self.kmeans.find_new_centroids(distances, idx_distances)

            self.t0 -= self.r
            counter += 1

        if candidate_solution == 1:
            print(f'Accuracy: {accuracy_score(self.kmeans.labels, swapped_clusters.reshape(-1, 1))}, '
                  f'precision: {precision_score(self.kmeans.labels, swapped_clusters.reshape(-1, 1), average="weighted")}'
                  f', recall: {recall_score(self.kmeans.labels, swapped_clusters.reshape(-1, 1), average="weighted")}')
            print(f'Confusion matrix: {confusion_matrix(self.kmeans.labels, swapped_clusters.reshape(-1, 1))}')
        elif candidate_solution == 0:
            print(f'Accuracy: {accuracy_score(self.kmeans.labels, clusters.reshape(-1, 1))}, '
                  f'precision: {precision_score(self.kmeans.labels, clusters.reshape(-1, 1), average="weighted")}'
                  f', recall: {recall_score(self.kmeans.labels, clusters.reshape(-1, 1), average="weighted")}')
            print(f'Confusion matrix: {confusion_matrix(self.kmeans.labels, clusters.reshape(-1, 1))}')


if __name__ == '__main__':
    sim_ann = SimulatedAnnealer(t0=250_000_000, tf=100_000_000, r=7_500_000, b=1200, k=5)
    sim_ann.run_iteration()