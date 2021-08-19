import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

rng = range(100)

def process_file(fin):
    with open(fin) as f:
        data = {}
        for line in f:
            line = line.strip()
            if line != "":
                l_split = line.split(",")
                if len(l_split) != 2:
                    continue
                fn, val = l_split
                if "both" in val.lower():
                    val = "Both"
                elif val.strip() == "Male" :
                    val = "Male"
                elif val.strip() == "Female":
                    val = "Female"
                else:
                    val = "Uncertain"
                if int(fn.replace(".jpg","").replace(".png","")) in rng:
                    data[fn] = val
                
    return data


if __name__ == "__main__":
    import sys

    ref_file = sys.argv[1]
    pred_file = sys.argv[2]

    d_pred = process_file(pred_file)
    d_ref = process_file(ref_file)

    dk1 = list(d_pred.keys())
    dk2 = list(d_ref.keys())

    dk1.sort()
    dk2.sort()

    pred = [d_pred[i] for i in dk1]
    ref = [d_ref[i] for i in dk1]

    ref_frequency = Counter(ref)
    pred_frequency = Counter(pred)

    print("=============================")
    print("REF Frequency", ref_frequency)
    print("Pred Frequency", pred_frequency)
    print("=============================")

    print("Overall results")
    print(precision_recall_fscore_support(ref, pred, average="micro"))
    print("=============================")

    print("Class wise results")
    label_list = list(ref_frequency.keys())
    prfs = precision_recall_fscore_support(ref, pred, average=None, labels=label_list)
    precision = prfs[0]
    recall = prfs[1]
    f1_scores = prfs[2]
    for i, label in enumerate(label_list):
        print(label + ":", precision[i], recall[i], f1_scores[i])
    print("=============================")
    print("Accuracy Score")
    print(accuracy_score(ref, pred))

    print("=============================")
    print("Confusion Matrix")
    df = pd.DataFrame(
        confusion_matrix(ref, pred, labels=label_list),
        index=label_list,
        columns=label_list,
    )
    print(df)
