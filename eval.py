"""Usage:
    eval.py <truth.csv> <preds.csv>"""
import sys
import csv


def load_csv(fn):
    res = {}
    for row in csv.DictReader(open(fn)):
        res[row['fn']] = row['label']
    return res


if len(sys.argv) < 3:
    sys.exit(__doc__)
truth = load_csv(sys.argv[1])
preds = load_csv(sys.argv[2])


correct = 0
wrong = 0
missing = 0
extra = 0


for fn, truth_label in truth.items():
    if fn not in preds:
        missing += 1
        continue

    if preds[fn] == truth_label:
        correct += 1
    else:
        wrong += 1


for fn, pred_label in preds.items():
    if fn not in truth:
        extra += 1

def fmt_pct(a, b):
    return "%6.2f%% (%5d/%5d)" % (a / (b + 1e-8) * 100, a, b)


print("Results:")
print("    Extra:       %s" % (extra,))
print("    Missing:     %s" % fmt_pct(missing, correct+wrong+missing))
print("    Predicted:   %s" % fmt_pct(correct+wrong+missing - missing, correct+wrong+missing))
print("        Correct: %s" % fmt_pct(correct, correct+wrong))
print("        Wrong:   %s" % fmt_pct(wrong, correct+wrong))

