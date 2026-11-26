"""Part 1: estimation, inference and regression baseline (all 800 rows)."""
import math
import numpy as np
from common import load_data, write_results, sample_sd

X, outcome, y = load_data()
x4 = X[:, 3]
n = len(y)
R = []  # (key, value) in output order


def add(k, v):
    R.append((k, v))


# 1.1 estimate with an interval
mean_value = float(y.sum() / n)
sd_value = sample_sd(y)
se_value = sd_value / math.sqrt(n)
add("n", n)
add("mean_value", mean_value)
add("sd_value", sd_value)
add("se_value", se_value)
add("ci95_low", mean_value - 1.96 * se_value)
add("ci95_high", mean_value + 1.96 * se_value)

# 1.2 z-test of H0: mu = 8.0
z_stat = (mean_value - 8.0) / se_value
add("z_stat", z_stat)
add("reject_h0", abs(z_stat) > 1.96)

# 1.3 required sample size for margin of error 0.5
need = (1.96 * sd_value / 0.5) ** 2
add("n_required", int(math.ceil(need)))


def simple_fit(x, t):
    """Closed-form simple regression, own code. Returns slope, intercept, fitted."""
    mx, mt = x.mean(), t.mean()
    cov = np.sum((x - mx) * (t - mt)) / (len(x) - 1)
    var = np.sum((x - mx) ** 2) / (len(x) - 1)
    slope = cov / var
    intercept = mt - slope * mx
    return slope, intercept, intercept + slope * x


# 1.4 simple regression, value on X4
slope, intercept, fitted = simple_fit(x4, y)
tss = float(np.sum((y - y.mean()) ** 2))
ess = float(np.sum((fitted - y.mean()) ** 2))
rss = float(np.sum((y - fitted) ** 2))
add("reg_slope", slope)
add("reg_intercept", intercept)
add("tss", tss)
add("ess", ess)
add("rss", rss)
add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add("r2", 1 - rss / tss)
add("rmse", math.sqrt(rss / n))
add("mae", float(np.mean(np.abs(y - fitted))))

# 1.5 gradient descent on standardised X4
mx4, sd4 = x4.mean(), sample_sd(x4)
z = (x4 - mx4) / sd4
m, c = 0.0, 0.0
eta = 0.1


def cost(m, c):
    e = y - (m * z + c)
    return float(np.sum(e ** 2) / n)


add("gd_J0", cost(m, c))
saved = {}
for step in range(1, 501):
    e = y - (m * z + c)
    dm = -(2.0 / n) * np.sum(z * e)
    dc = -(2.0 / n) * np.sum(e)
    m -= eta * dm
    c -= eta * dc
    if step in (1, 10, 100, 500):
        saved[step] = cost(m, c)
for s in (1, 10, 100, 500):
    add("gd_J%d" % s, saved[s])
gd_slope = m / sd4
gd_intercept = c - m * mx4 / sd4
add("gd_slope", gd_slope)
add("gd_intercept", gd_intercept)
add("gd_slope_gap", abs(gd_slope - slope))


# 1.6 funnel check and log remedy
def halves_ratio(resid, fit):
    order = np.argsort(fit, kind="stable")
    lo = sample_sd(resid[order[:400]])
    hi = sample_sd(resid[order[400:]])
    return lo, hi, hi / lo


lo, hi, ratio = halves_ratio(y - fitted, fitted)
add("resid_sd_lower_raw", lo)
add("resid_sd_upper_raw", hi)
add("resid_sd_ratio_raw", ratio)

ly, lx4 = np.log(y), np.log(x4)
ll_slope, ll_int, ll_fit = simple_fit(lx4, ly)
ll_rss = float(np.sum((ly - ll_fit) ** 2))
ll_tss = float(np.sum((ly - ly.mean()) ** 2))
ll_r2 = 1 - ll_rss / ll_tss
add("loglog_intercept", ll_int)
add("loglog_slope", ll_slope)
add("loglog_r2", ll_r2)
add("loglog_rmse", math.sqrt(ll_rss / n))
lo, hi, ratio = halves_ratio(ly - ll_fit, ll_fit)
add("resid_sd_lower_log", lo)
add("resid_sd_upper_log", hi)
add("resid_sd_ratio_log", ratio)

# 1.7 multiple regression by the normal equation
D = np.column_stack([np.ones(n), lx4, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(D.T @ D, D.T @ ly)
for i in range(8):
    add("mr_beta_%d" % i, beta[i])
mr_rss = float(np.sum((ly - D @ beta) ** 2))
mr_r2 = 1 - mr_rss / ll_tss
p = 7
add("mr_r2", mr_r2)
add("mr_adj_r2", 1 - (1 - mr_r2) * (n - 1) / (n - p - 1))
add("loglog_adj_r2", 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1))

write_results("part1_results.txt", R)
