"""Part 4: model choice (splits, voting, AdaBoost by hand, oob share, network sizing)."""
import math
import numpy as np
from common import write_results, write_csv
from models import (get_split, train_logistic, predict_proba, confusion, metrics,
                    knn_neighbour_share, choose_knn_k)

S = get_split()
Xtr, Xte, ytr, yte = S["Xtr"], S["Xte"], S["ytr"], S["yte"]
Rtr, Rte = S["X_raw_tr"], S["X_raw_te"]          # original units (columns X1..X7 = 0..6)
R = []


def add(k, v):
    R.append((k, v))


def entropy(pos, n):
    if n == 0:
        return 0.0
    out = 0.0
    for p in (pos / n, 1 - pos / n):
        if p > 0:
            out -= p * math.log2(p)
    return out


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p ** 2 - (1 - p) ** 2


# 4.1 which split is purest (training rows, original units)
n = len(ytr)
P = int(ytr.sum())
add("root_entropy", entropy(P, n))
add("root_gini", gini(P, n))
conds = {
    "S1": lambda R_: R_[:, 5] == 1,
    "S2": lambda R_: R_[:, 6] == 1,
    "S3": lambda R_: R_[:, 2] > 0.5,
    "S4": lambda R_: R_[:, 0] <= 24,
}
split_rows, gains = [], {}
for name, f in conds.items():
    yes = f(Rtr)
    ny, py = int(yes.sum()), int(ytr[yes].sum())
    nn, pn = int((~yes).sum()), int(ytr[~yes].sum())
    ig = entropy(P, n) - (ny / n * entropy(py, ny) + nn / n * entropy(pn, nn))
    gg = gini(P, n) - (ny / n * gini(py, ny) + nn / n * gini(pn, nn))
    split_rows.append((name, ny, py, nn, pn, ig, gg))
    gains[name] = (ig, gg)
write_csv("part4_splits.csv", "split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain", split_rows)
best_ig = max(gains, key=lambda s: gains[s][0])
best_gg = max(gains, key=lambda s: gains[s][1])
add("best_split_by_info_gain", best_ig)
add("best_split_by_gini_gain", best_gg)

# 4.2 stump + voting
yes_tr = conds[best_ig](Rtr)
rate_yes = float(ytr[yes_tr].sum() / yes_tr.sum())
rate_no = float(ytr[~yes_tr].sum() / (~yes_tr).sum())
add("stump_rate_yes", rate_yes)
add("stump_rate_no", rate_no)

w, b, _ = train_logistic(Xtr, ytr, 0.1, 1000)
_, best_k = choose_knn_k(Xtr, ytr)
p_log = predict_proba(Xte, w, b)
p_knn = knn_neighbour_share(Xtr, ytr, Xte, best_k)
p_stump = np.where(conds[best_ig](Rte), rate_yes, rate_no)
EPS = 1e-12
pred_log = p_log >= 0.3 - EPS
pred_knn = p_knn * best_k >= (best_k + 1) / 2 - EPS
pred_stump = p_stump >= 0.3 - EPS
votes = pred_log.astype(int) + pred_knn.astype(int) + pred_stump.astype(int)
pred_hard = votes >= 2
pred_soft = (p_log + p_knn + p_stump) / 3 >= 0.3 - EPS
vote_rows = []
for name, pred in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", pred_hard), ("soft_vote", pred_soft)):
    tp, fp, tn, fn = confusion(pred, yte)
    m = metrics(tp, fp, tn, fn)
    vote_rows.append((name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]))
write_csv("part4_voting.csv", "model,tp,fp,tn,fn,precision,recall,f1", vote_rows)

# 4.3 two rounds of AdaBoost by hand
y8 = np.array([+1, +1, +1, -1, -1, -1, +1, -1])
h1 = np.array([+1, +1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, +1, +1, -1, +1, -1, +1, -1])
w_now = np.full(8, 1 / 8)
alphas, ada_rows = [], []
for rnd, h in ((1, h1), (2, h2)):
    eps = float(w_now[h != y8].sum())
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = w_now * np.exp(-alpha * y8 * h)
    w_next = unnorm / unnorm.sum()
    for pt in range(8):
        ada_rows.append((rnd, pt + 1, w_now[pt], unnorm[pt], w_next[pt]))
    add("round%d_error" % rnd, eps)
    add("round%d_alpha" % rnd, alpha)
    alphas.append(alpha)
    w_now = w_next
write_csv("part4_adaboost.csv", "round,point,weight_before,weight_unnormalised,weight_after", ada_rows)
final = np.sign(alphas[0] * h1 + alphas[1] * h2)
add("ensemble_accuracy", float(np.mean(final == y8)))

# 4.4 out-of-bag share
add("oob_fraction", (1 - 1 / 560) ** 560)


# 4.5 size the networks (nothing is trained)
def conv_out(size, f, pad, stride=1):
    return (size - f + 2 * pad) // stride + 1


mlp1 = 784 * 16 + 16
mlp2 = 16 * 16 + 16
mlp3 = 16 * 10 + 10
s1 = conv_out(28, 5, 2)                 # conv1 output side
p1 = conv_out(s1, 2, 0, 2)              # after max-pool
s2 = conv_out(p1, 5, 0)                 # conv2 output side
p2 = conv_out(s2, 2, 0, 2)              # after max-pool
flat = p2 * p2 * 16
cnn1 = 5 * 5 * 1 * 6 + 6
cnn2 = 5 * 5 * 6 * 16 + 16
cnn3 = flat * 32 + 32
cnn4 = 32 * 10 + 10
print("CNN shapes: conv1 %dx%dx6, pool %dx%dx6, conv2 %dx%dx16, pool %dx%dx16, flatten %d"
      % (s1, s1, p1, p1, s2, s2, p2, p2, flat))
write_csv("part4_architecture.csv", "layer,parameters", [
    ("mlp_hidden1", mlp1), ("mlp_hidden2", mlp2), ("mlp_output", mlp3),
    ("cnn_conv1", cnn1), ("cnn_conv2", cnn2), ("cnn_dense1", cnn3), ("cnn_dense2", cnn4)])
add("mlp_total", mlp1 + mlp2 + mlp3)
add("cnn_total", cnn1 + cnn2 + cnn3 + cnn4)
add("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
add("conv_layer_params", 3 * 3 * 3 * 64 + 64)
add("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)

write_results("part4_results.txt", R)
