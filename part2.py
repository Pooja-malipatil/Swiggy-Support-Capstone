"""Part 2: classification and evaluation (logistic regression, KNN, threshold by cost)."""
import math
import numpy as np
from common import write_results, write_csv
from models import (get_split, train_logistic, bce, predict_proba, confusion, metrics,
                    knn_predict, choose_knn_k)

S = get_split()
Xtr, Xte, ytr, yte = S["Xtr"], S["Xte"], S["ytr"], S["yte"]
R = []


def add(k, v):
    R.append((k, v))


# 2.1 split sizes
add("train_rows", len(ytr))
add("train_positives", int(ytr.sum()))
add("test_rows", len(yte))
add("test_positives", int(yte.sum()))

# 2.2 logistic regression from scratch
w, b, losses = train_logistic(Xtr, ytr, 0.1, 1000, record=(1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    add("loss_epoch_%d" % e, losses[e])
for j in range(7):
    add("weight_X%d" % (j + 1), w[j])
add("bias", b)
add("test_bce", bce(Xte, yte, w, b))

# 2.3 learning-rate effect
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    add("loss_epoch_200_eta_%s" % eta, l[200])

# 2.4 odds ratios
for j in range(7):
    add("odds_ratio_X%d" % (j + 1), math.exp(w[j]))

# 2.5 two thresholds on the test rows
p_te = predict_proba(Xte, w, b)
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(p_te >= tau, yte)
    m = metrics(tp, fp, tn, fn)
    for key, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", m["accuracy"]),
                     ("precision", m["precision"]), ("recall", m["recall"]), ("f1", m["f1"]),
                     ("specificity", m["specificity"])):
        add("tau_%s_%s" % (tau, key), val)
add("blind_accuracy", float(np.sum(yte == 0) / len(yte)))

# 2.6 threshold sweep
rows = []
for i in range(1, 10):
    tau = i / 10
    tp, fp, tn, fn = confusion(p_te >= tau - 1e-12, yte)
    m = metrics(tp, fp, tn, fn)
    rows.append((tau, tp, fp, tn, fn, m["recall"], m["fpr"]))
write_csv("part2_thresholds.csv", "tau,tp,fp,tn,fn,tpr,fpr", rows)

# 2.7 KNN with 5-fold cross-validation on the training rows
cv_rows, best_k = choose_knn_k(Xtr, ytr)
write_csv("part2_knn_cv.csv", "K,mean_cv_f1", cv_rows)
add("knn_best_k", best_k)
pred, _ = knn_predict(Xtr, ytr, Xte, best_k)
tp, fp, tn, fn = confusion(pred, yte)
m = metrics(tp, fp, tn, fn)
for key, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", m["accuracy"]),
                 ("precision", m["precision"]), ("recall", m["recall"]), ("f1", m["f1"])):
    add("knn_%s" % key, val)

# 2.8 choose the alert threshold with money (training rows only)
p_tr = predict_proba(Xtr, w, b)


def cost(p, y, tau):
    _, fp, _, fn = confusion(p >= tau - 1e-12, y)
    return 50 * fp + 400 * fn


costs = [(k / 20, cost(p_tr, ytr, k / 20)) for k in range(1, 20)]
cost_tau, cost_train = min(costs, key=lambda r: (r[1], r[0]))
add("cost_tau", cost_tau)
add("cost_train_at_tau", cost_train)
add("cost_test_at_tau", cost(p_te, yte, cost_tau))
add("cost_test_at_0.5", cost(p_te, yte, 0.5))
add("cost_test_flag_nobody", 400 * int(yte.sum()))

write_results("part2_results.txt", R)
