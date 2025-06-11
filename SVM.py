import numpy as np
import pandas as pd
from sklearn import svm
from collections import Counter, defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

'''
Balanced accuracy

Kernel matrices calculated using k-mer similarity

Data:
- K=180 training samples
- K=56 test samples
- 19 classes - very imbalanced

Experiments:
- Only 5 largest classes in training/test
- Only 10 largest classes in training/test
- All 19 classes
For each experiment, repeat 20 times with different
random seeds and report:
- Mean, STD, Balanced Accuracy
- C.class_weight = balanced
- Precomputed kernel
'''

def experiment(train_kernel, train_labels, test_kernel, test_labels, top_n, seed=42):
    if top_n is not None:
        class_counts = Counter(train_labels)
        top_classes = set([cls for cls, _ in class_counts.most_common(top_n)])

        train_idx = [i for i, c in enumerate(train_labels) if c in top_classes]
        test_idx = [i for i, c in enumerate(test_labels) if c in top_classes]

        train_kernel = train_kernel[train_idx,:][:,train_idx]
        train_labels = [train_labels[i] for i in train_idx]

        test_kernel = test_kernel[test_idx,:][:, train_idx]
        test_labels = [test_labels[i] for i in test_idx]

    classifier = svm.SVC(kernel='precomputed', class_weight='balanced',random_state=seed)
    classifier.fit(train_kernel,train_labels)

    y_pred = classifier.predict(test_kernel)

    return classification_report(test_labels, y_pred), balanced_accuracy_score(test_labels, y_pred)

def load_kernels(tf):
    df = pd.read_csv(tf, sep="\t", index_col=0)
    kernel = df.values
    labels = df.index.tolist()
    return kernel, labels

def load_labels(tf):
    label_map = {}
    with open(tf, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            _, label, sample_id = parts
            label_map[sample_id] = label
    return label_map

if __name__ == "__main__":
    training_kernel, training_ids = load_kernels('data/training_kernel_matrix.txt')
    training_labels = load_labels('data/training_labels.txt')
    y_train = [training_labels[sample_id] for sample_id in training_ids]


    t2t_kernel_matrix, t2t_ids = load_kernels('data/test2training_kernel_matrix.txt')
    test_labels = load_labels('data/test_labels.txt')
    y_test = [test_labels[sample_id] for sample_id in t2t_ids]

    seeds = [i for i in range(20)]
    reports = defaultdict(list)
    results = defaultdict(list)

    for top_n in [5, 10, None]:
        for seed in seeds:#20 random seeds
            report, acc = experiment(training_kernel, y_train, t2t_kernel_matrix, y_test, top_n=top_n, seed=seed)
            reports[str(top_n)].append(report)
            results[str(top_n)].append(acc)
    for key in results:
        accs = results[key]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        label = f"Top {key} largest classes" if key != "None" else "All Classes"
        print(f"{label} - Mean Balanced Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")