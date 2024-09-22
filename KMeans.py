# knn3
#lc0.1
# sc10rbf
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)


n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_flat)

train_clusters = kmeans.predict(X_train_flat)
test_clusters = kmeans.predict(X_test_flat)

cluster_labels = np.zeros(n_clusters, dtype=int)
for i in range(n_clusters):
    mask = (train_clusters == i)
    cluster_labels[i] = mode(y_train[mask])[0]

models = {
    'logistic_regression': LogisticRegression(max_iter=500),
    'knn': KNeighborsClassifier(),
    'svc': SVC()
}

trained_models = {model_name: {} for model_name in models}

for model_name, model in models.items():
    print(f"Training models with {model_name}")

    for cluster_id in range(n_clusters):
        
        X_cluster = X_train_flat[train_clusters == cluster_id]
        y_cluster = y_train[train_clusters == cluster_id]

        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)

        model_clone = model
        model_clone.fit(X_train_split, y_train_split)
        trained_models[model_name][cluster_id] = model_clone


param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
knn_grid.fit(X_train_flat, y_train)
print(f"Best parameters for KNN: {knn_grid.best_params_}")


param_grid_logistic = {'C': [0.1, 1, 10, 100]}
logistic_grid = GridSearchCV(LogisticRegression(max_iter=500), param_grid_logistic, cv=5)
logistic_grid.fit(X_train_flat, y_train)
print(f"Best parameters for Logistic Regression: {logistic_grid.best_params_}")

param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svc_grid = GridSearchCV(SVC(), param_grid_svc, cv=5)
svc_grid.fit(X_train_flat, y_train)
print(f"Best parameters for SVC: {svc_grid.best_params_}")

for model_name, model_dict in trained_models.items():
    print(f"\nEvaluating models for {model_name}")

    for cluster_id in range(n_clusters):
        model = model_dict[cluster_id]

        X_test_cluster = X_test_flat[test_clusters == cluster_id]
        y_test_cluster = y_test[test_clusters == cluster_id]

        if len(X_test_cluster) > 0:
           
            y_pred = model.predict(X_test_cluster)

            y_pred_mapped = cluster_labels[y_pred]

            f1 = f1_score(y_test_cluster, y_pred_mapped, average='micro')
            accuracy = accuracy_score(y_test_cluster, y_pred_mapped)
            precision = precision_score(y_test_cluster, y_pred_mapped, average='micro')
            recall = recall_score(y_test_cluster, y_pred_mapped, average='micro')

            print(f'F1 Score for Cluster {cluster_id} with {model_name}: {f1}')
            print(f'Accuracy for Cluster {cluster_id} with {model_name}: {accuracy}')
            print(f'Precision for Cluster {cluster_id} with {model_name}: {precision}')
            print(f'Recall for Cluster {cluster_id} with {model_name}: {recall}')
