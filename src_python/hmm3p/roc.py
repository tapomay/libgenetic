import math
import matplotlib.pyplot as plt
from texttable import Texttable

def load(suffix = ''):
    with open('positives%s' % suffix) as f:
        positives = f.read()
        positives = positives.split()
        positives = [float(v) for v in positives]
    with open('negatives%s' % suffix) as f:
        negatives = f.read()
        negatives = negatives.split()
        negatives = [float(v) for v in negatives]
    return (positives, negatives)

def roc_compute(positives, negatives):
    minThresh = math.floor(min(positives + negatives))
    maxThresh = math.ceil(max(positives + negatives))

    threshDelta = (maxThresh - minThresh) / 10
    thresh = minThresh
    fpr_X = []
    tpr_Y = []
    vals = []
    vals.append(["Thresh", "TP", "FP", "TN", "FN", "FPR", "TPR"])
    while thresh < maxThresh:
        tp = len([v for v in positives if v >= thresh])
        fn = len([v for v in positives if v < thresh])
        tn = len([v for v in negatives if v <= thresh])
        fp = len([v for v in negatives if v > thresh])
        fpr = fp / float(fp + tn) * 100 if (fp+tn) != 0 else -1
        tpr = tp / float(tp + fn) * 100 if (tp+fn) != 0 else -1
        if fpr !=-1 and tpr !=-1:
            stat = [thresh, tp, fp, tn, fn, fpr, tpr]
            print("thresh: %f, tp:%d, fp:%d, tn:%d, fn:%d, fpr:%f, tpr: %f" % (thresh, tp, fp, tn, fn, fpr, tpr))
            fpr_X.append(fpr)
            tpr_Y.append(tpr)
            # ROC = fpr@X vz tpr#Y
            vals.append(stat)
        thresh += threshDelta

    return (fpr_X, tpr_Y, vals)


def main():
    (positives, negatives) = load('_4')
    print("POS: %s" % positives)
    print("NEG: %s" % negatives)

    (fpr_X, tpr_Y, vals) = roc_compute(positives, negatives)
    t = Texttable()

    for v in vals:
        t.add_row(v)
    print(t.draw())
    # plot
    plt.plot(fpr_X, tpr_Y)
    plt.ylabel('TPR = TP/(TP + FN)')
    plt.xlabel('FPR = FP/(FP+TN)')
    plt.axis([0, 100, 0, 100])
    plt.show()

if __name__ == "__main__":
    main()
