import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ── PREPARE DATA ──────────────────────────────────────────────
def prepare_clustering_data(df):
    df = df.copy()
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        return None, None
    df = df.dropna()
    if len(df) < 5:
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return df, scaled_data


# ── ELBOW METHOD ──────────────────────────────────────────────
def find_optimal_clusters(data, max_k=10):
    inertia = []
    K = range(2, min(max_k, len(data)))
    for k in K:
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(data)
            inertia.append(model.inertia_)
        except Exception:
            inertia.append(None)
    return list(K), inertia


# ── SAFE SILHOUETTE ───────────────────────────────────────────
def safe_silhouette(data, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if len(unique_labels) < 2:
        return -1
    # Filter out noise points for scoring
    mask = np.array(labels) != -1
    if mask.sum() < 2:
        return -1
    try:
        return silhouette_score(data[mask], np.array(labels)[mask])
    except Exception:
        return -1


# ── KMEANS (auto-best K via silhouette sweep) ─────────────────
def run_kmeans(data, n_clusters=None):
    n = len(data)
    max_k = min(10, n - 1)

    if n_clusters is not None and n_clusters >= 2:
        # Use provided k directly
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data)
        score = safe_silhouette(data, labels)
        return labels, score

    # Auto-sweep k=2..max_k, pick best silhouette
    best_k, best_score, best_labels = 2, -1, None
    for k in range(2, max(3, max_k + 1)):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = model.fit_predict(data)
            sc = safe_silhouette(data, lbl)
            if sc > best_score:
                best_score = sc
                best_k = k
                best_labels = lbl
        except Exception:
            pass

    if best_labels is None:
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = model.fit_predict(data)
        best_score = safe_silhouette(data, best_labels)

    return best_labels, best_score


# ── DBSCAN (auto-eps via k-distance heuristic) ────────────────
def run_dbscan(data, eps=None, min_samples=None):
    n = len(data)
    n_features = data.shape[1] if len(data.shape) > 1 else 1

    # Auto min_samples
    if min_samples is None:
        min_samples = max(2, min(5, n // 20))

    # Auto eps via nearest-neighbor distances
    if eps is None:
        try:
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data)
            distances, _ = nbrs.kneighbors(data)
            k_distances = np.sort(distances[:, -1])
            # Use 90th percentile as eps (elbow approximation)
            eps = float(np.percentile(k_distances, 90))
            eps = max(0.1, min(eps, 5.0))  # clamp to sane range
        except Exception:
            eps = 0.5

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    # If all noise or all same cluster, try relaxing eps
    unique = set(labels)
    if len(unique - {-1}) < 2:
        for factor in [1.5, 2.0, 3.0]:
            try:
                m2 = DBSCAN(eps=eps * factor, min_samples=max(2, min_samples - 1))
                lbl2 = m2.fit_predict(data)
                if len(set(lbl2) - {-1}) >= 2:
                    labels = lbl2
                    break
            except Exception:
                pass

    score = safe_silhouette(data, labels)
    return labels, score


# ── AGGLOMERATIVE (auto-best K via silhouette sweep) ──────────
def run_agglomerative(data, n_clusters=None):
    n = len(data)
    max_k = min(10, n - 1)

    if n_clusters is not None and n_clusters >= 2:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(data)
        score = safe_silhouette(data, labels)
        return labels, score

    best_k, best_score, best_labels = 2, -1, None
    for k in range(2, max(3, max_k + 1)):
        try:
            model = AgglomerativeClustering(n_clusters=k)
            lbl = model.fit_predict(data)
            sc = safe_silhouette(data, lbl)
            if sc > best_score:
                best_score = sc
                best_k = k
                best_labels = lbl
        except Exception:
            pass

    if best_labels is None:
        model = AgglomerativeClustering(n_clusters=2)
        best_labels = model.fit_predict(data)
        best_score = safe_silhouette(data, best_labels)

    return best_labels, best_score


# ── PCA FOR VISUALIZATION ─────────────────────────────────────
def reduce_to_2d(data):
    try:
        n_comp = min(2, data.shape[1] if len(data.shape) > 1 else 1)
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(data)
        if reduced.shape[1] < 2:
            # Pad with zeros if only 1 component
            reduced = np.hstack([reduced, np.zeros((reduced.shape[0], 1))])
        return reduced
    except Exception:
        return None


# ── AUTO CLUSTERING ───────────────────────────────────────────
def run_all_clustering(data):
    results = {}
    for name, func in {
        "KMeans": run_kmeans,
        "DBSCAN": run_dbscan,
        "Agglomerative": run_agglomerative
    }.items():
        try:
            labels, score = func(data)
            results[name] = {
                "labels": labels,
                "score": round(score, 4)
            }
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results[name] = {"labels": None, "score": -1}
    return results


# ── BEST CLUSTER ──────────────────────────────────────────────
def get_best_clustering(results):
    best_model = None
    best_score = -1
    for name, res in results.items():
        if res["score"] > best_score:
            best_score = res["score"]
            best_model = name
    return best_model, best_score