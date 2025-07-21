import numpy as np
from itertools import chain, repeat
from sklearn.metrics import f1_score
from scipy.special import softmax
from scipy.spatial.distance import cdist


class KNNGraphClassifier:
    def __init__(self, dataset, train_idx, test_idx, node_embeddings):
        self.dataset = dataset
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.node_embeddings = node_embeddings

        # Flatten node embeddings
        self.training_embeddings = np.vstack([node_embeddings[i] for i in train_idx])
        self.testing_embeddings = np.vstack([node_embeddings[i] for i in test_idx])

        # Graph-level labels repeated per node
        self.training_labels = np.repeat(
            [dataset[i].y.item() for i in train_idx],
            [dataset[i].num_nodes for i in train_idx]
        )

        self.testing_indices = np.array(
            list(chain.from_iterable(
                repeat(i, dataset[i].num_nodes) for i in test_idx
            ))
        )

        self.test_graph_ids = test_idx

        self.distances = cdist(self.testing_embeddings, self.training_embeddings)
    
    def predict(self, k=1, weighted=False):
        if weighted:
            return self._predict_weighted_knn(k)
        else:
            return self._predict_knn(k)
    
    def _predict_knn(self, k):
        topk_indices = np.argpartition(self.distances, kth=range(k), axis=1)[:, :k]
        topk_labels = self.training_labels[topk_indices]

        node_preds = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.training_labels.max() + 1).argmax(),
            axis=1,
            arr=topk_labels
        )

        return self._aggregate_node_predictions(node_preds)
    
    def _predict_weighted_knn(self, k):
        topk_indices = np.argpartition(self.distances, kth=range(k), axis=1)[:, :k]
        topk_dists = np.take_along_axis(self.distances, topk_indices, axis=1)
        topk_labels = self.training_labels[topk_indices]

        weights = softmax(-topk_dists, axis=1)
        num_classes = self.training_labels.max() + 1
        label_probs = np.zeros((topk_labels.shape[0], num_classes), dtype=np.float32)

        for i in range(k):
            np.add.at(label_probs, (np.arange(topk_labels.shape[0]), topk_labels[:, i]), weights[:, i])

        node_preds = label_probs.argmax(axis=1)
        return self._aggregate_node_predictions(node_preds)

    def _aggregate_node_predictions(self, node_preds):
        graph_pred_map = {}
        for graph_id in self.test_graph_ids:
            mask = self.testing_indices == graph_id
            if not np.any(mask):
                continue
            graph_node_preds = node_preds[mask]
            graph_pred = np.argmax(np.bincount(graph_node_preds))
            graph_pred_map[graph_id] = graph_pred
        return [graph_pred_map[g] for g in self.test_graph_ids]

    def evaluate(self, predicted_labels):
        true_labels = [self.dataset[i].y.item() for i in self.test_graph_ids]
        return f1_score(true_labels, predicted_labels)
