"""Shared helpers. Importing this file does not run any Part."""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def load_data():
    """Return (X, outcome, value): X is 800 x 7 (X1..X7), outcome and value are length-800 arrays."""
    with open(os.path.join(HERE, "data", "records.csv"), newline="") as f:
        rows = list(csv.reader(f))[1:]
    arr = np.array([[float(v) for v in r[1:]] for r in rows])
    return arr[:, 0:7], arr[:, 7].astype(int), arr[:, 8]


def fmt(v):
    """Results-file format: 4 decimals for floats, plain integers, yes/no for booleans."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return "%.4f" % float(v)


def write_results(name, pairs):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        for k, v in pairs:
            f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def sample_sd(a):
    return float(np.sqrt(np.sum((a - a.mean()) ** 2) / (len(a) - 1)))
