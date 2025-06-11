from sklearn import svm
import numpy as np

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

def experiment(seed=42):
    classifier = svm.SVC(kernel='precomputed', class_weight='balanced',random_state=seed)
    pass

def load_labels(tf):
    labels = {}
    with open(tf, 'r') as f:
        for line in f:
            _, name, id = line.split("\t")
            id = id.strip().replace("\n","")
            try:
                labels[name].append(id)
            except:
                labels[name] = [id]
    return labels

if __name__ == "__main__":
    training_labels = load_labels('data/training_labels.txt')
    test_labels = load_labels('data/test_labels.txt')
    
    training_kernel_matrix = np.loadtxt('data/training_kernel_matrix.txt')
    test2training_kernel_matrix = np.loadtxt('data/test2training_kernel_matrix.txt')
    
    experiment()