"""Part 3: segmentation (K-means, hierarchical clustering) and PCA, all 800 rows, X1..X5."""
import numpy as np
from common import load_data, write_results, write_csv, sample_sd

X, outcome, value = load_data()
X5 = X[:, :5]
n = len(X5)
Z = (X5 - X5.mean(axis=0)) / np.array([sample_sd(X5[:, j]) for j in range(5)])
R = []


def add(k, v):
    R.append((k, v))


def kmeans(Z, K):
    """Pinned K-means: start at rows 0, n//K, 2*(n//K), ...; stop at the first pass with no change."""
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = np.full(len(Z), -1)
    passes = 0
    while True:
        d2 = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)          # argmin picks the lower-numbered centroid on ties
        passes += 1
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(K):
            members = Z[labels == c]
            if len(members):                 # an empty cluster keeps its old centroid
                cent[c] = members.mean(axis=0)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, wcss, passes


# 3.1 K-means for K = 2..6
wcss, results = {}, {}
for K in range(2, 7):
    lab, w, passes = kmeans(Z, K)
    wcss[K] = w
    results[K] = lab
    add("wcss_k%d" % K, w)
    print("K=%d passes=%d" % (K, passes))

# 3.2 elbow rule
elbow_k = None
for K in (3, 4, 5):
    if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
        elbow_k = K
        break
add("elbow_k", elbow_k if elbow_k is not None else 0)

# 3.3 profile the K = 3 segments, ordered by ascending mean X3 (original units)
lab3 = results[3]
order = sorted(range(3), key=lambda c: X[lab3 == c, 2].mean())
seg_of = np.zeros(n, dtype=int)
seg_rows = []
for name, c in zip("ABC", order):
    mask = lab3 == c
    seg_of[mask] = "ABC".index(name)
    seg_rows.append((name, int(mask.sum()), X[mask, 0].mean(), X[mask, 1].mean(),
                     X[mask, 2].mean(), X[mask, 3].mean(), X[mask, 4].mean(),
                     outcome[mask].mean(), value[mask].mean()))
    add("size_segment_%s" % name, int(mask.sum()))
write_csv("part3_segments.csv",
          "segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value", seg_rows)

# 3.4 hierarchical clustering on the supplied dissimilarity table
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def agglomerate(D, rule):
    clusters = [(i,) for i in range(6)]      # members as 0-based indices
    merges = []

    def link(a, b):
        dist = [D[i, j] for i in a for j in b]
        return min(dist) if rule == "single" else (max(dist) if rule == "complete" else sum(dist) / len(dist))

    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = link(clusters[i], clusters[j])
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = clusters[i], clusters[j]
        if min(a) > min(b):
            a, b = b, a
        merges.append((a, b, h))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [tuple(sorted(a + b))]
    return merges


def name(c):
    return "+".join("E%d" % (m + 1) for m in sorted(c))


hier_rows = []
for rule in ("single", "complete", "average"):
    merges = agglomerate(D, rule)
    for no, (a, b, h) in enumerate(merges, 1):
        hier_rows.append((rule, no, name(a), name(b), h))
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    below = int(np.argmax(gaps)) + 1          # merges below the cut
    add("%s_clusters_at_cut" % rule, 6 - below)
    print(rule, [(name(a), name(b), round(h, 3)) for a, b, h in merges], "cut ->", 6 - below)
write_csv("part3_hierarchical.csv", "linkage,merge_no,cluster_a,cluster_b,height", hier_rows)

# 3.5 PCA by the seven steps
mean_vec = Z.mean(axis=0)                      # step 1: ~0 after standardising
Xc = Z - mean_vec                              # step 2: centre
S = Xc.T @ Xc / (n - 1)                        # step 3: covariance matrix
vals, vecs = np.linalg.eigh(S)                 # step 4: eigen-decomposition
idx = np.argsort(vals)[::-1]                   # step 5: largest eigenvalue first
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):                             # sign convention: largest |entry| positive
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
W = vecs[:, :2]                                # step 6: top two eigenvectors
proj = Xc @ W                                  # step 7: project
total = float(vals.sum())
add("eigenvalue_sum", total)
for j in range(5):
    add("eigenvalue_%d" % (j + 1), vals[j])
for j in range(5):
    add("explained_ratio_%d" % (j + 1), vals[j] / total)
cum = np.cumsum(vals) / total
for j in range(5):
    add("cumulative_ratio_%d" % (j + 1), cum[j])
add("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for pc in (1, 2):
    for j in range(5):
        add("pc%d_loading_X%d" % (pc, j + 1), vecs[j, pc - 1])
for r in range(3):
    add("row%d_pc1" % (r + 1), proj[r, 0])
    add("row%d_pc2" % (r + 1), proj[r, 1])
for s, nm in enumerate("ABC"):
    add("mean_pc1_segment_%s" % nm, proj[seg_of == s, 0].mean())

write_results("part3_results.txt", R)
