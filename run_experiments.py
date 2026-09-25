import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def _load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads data from the churn-data data set

    Returns
    -------
    X, y and feature names
    """
    data = np.genfromtxt("churn-data.csv", delimiter=",", dtype=float, names=True)
    feature_names = list(data.dtype.names[:-1])
    target_name = data.dtype.names[-1]
    X = np.array([data[feature] for feature in feature_names]).T
    y = data[target_name].astype(int)

    # print(f"Feature columns names: {feature_names}")
    # print(f"Target column name: {target_name}")
    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")

    return X, y, feature_names

def _split_data(
        X: np.ndarray, 
        y: np.ndarray,
        seed: int=42
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data in train and test data with spesified random state (seed)

    Returns
    --------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return X_train, X_test, y_train, y_test

def _find_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_class: Any=DecisionTree,
        seed: int=42,
        **kwargs
        ) -> tuple[str, int | None, dict]:
    """
    Finds optimal hyperparameters using k-fold cross validation

    Returns
    ------
    Best criterion
    max_depth
    Results
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed,)

    criterions = ["entropy", "gini"]
    depths = [1,2,3,4,5,6,7,10,None]

    best_score: float | None = None
    best_criterion: str
    best_max_depth: int | None = None
    results = {}
    
    # nested loop for testing all combinations of depths and criterion
    for criterion in criterions:
        results[criterion] = {}

        for depth in depths:
            val_scores = []
            train_scores = []
            unlimited_depths = []

            for train_index, val_index in kf.split(X_train):
                model = model_class(criterion=criterion, max_depth=depth, **kwargs)   
                X_train_fold = X_train[train_index]
                X_val_fold = X_train[val_index]

                y_train_fold = y_train[train_index]
                y_val_fold = y_train[val_index]

                model.fit(X_train_fold, y_train_fold)

                y_train_pred = model.predict(X_train_fold)
                y_val_pred = model.predict(X_val_fold)

                train_score = f1_score(y_train_fold, y_train_pred)
                val_score = f1_score(y_val_fold, y_val_pred)

                train_scores.append(train_score)
                val_scores.append(val_score)

                if depth is None:
                    unlimited_depths.append(model.get_depth())

            train_score = float(np.mean(train_scores))
            val_score = float(np.mean(val_scores))
            if depth is None:
                depth = int(round(np.mean(unlimited_depths)))

            results[criterion][depth] = {
                "train_f1": train_score,
                "val_f1": val_score
            }

            if best_score is None or val_score > best_score:
                best_score = val_score
                best_criterion = criterion
                best_max_depth = depth

    results["best"] = {
        "criterion": best_criterion,
        "max_depth": best_max_depth
    }
    
    return best_criterion, best_max_depth, results

def _test_different_seeds(
        X: np.ndarray,
        y: np.ndarray,
        criterion: str,
        max_depth: int|None,
        model_class: Any=DecisionTree,
        **kwargs
        ) -> tuple[float, float, float, float]:
    """Returns mean and standard deviation of scores over different random states"""
    random_states = [1,5,10,20,40,50,80,100,123,150] # 10 seeds
    f1_scores = []
    accuracy_scores = []

    for random_state in random_states:
        X_train, X_test, y_train, y_test = _split_data(X,y,seed=random_state)

        model = model_class(criterion=criterion,max_depth=max_depth, **kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    f1 = np.mean(f1_scores)
    f1_diff = np.std(f1_scores, ddof=1)
    accuracy = np.mean(accuracy_scores)
    accuracy_diff = np.std(accuracy_scores, ddof=1)
    return f1, f1_diff, accuracy, accuracy_diff

def run_pipeline():
    X, y, _ = _load_data()
    X_train, _, y_train, _ = _split_data(X, y)
    criterion, max_depth, result = _find_hyperparameters(X_train, y_train)
    sk_criterion, sk_max_depth, sk_result = _find_hyperparameters(
        X_train,
        y_train,
        model_class=DecisionTreeClassifier,
        random_state = 42,
        class_weight="balanced"
        )
    
    
    f1, f1_diff, acc, acc_diff = _test_different_seeds(X, y, criterion, max_depth)
    sk_f1, sk_f1_diff, sk_acc, sk_acc_diff= _test_different_seeds(
        X,
        y,
        sk_criterion,
        sk_max_depth,
        model_class=DecisionTreeClassifier,
        random_state = 42,
        class_weight="balanced"
        )

    print("Results")
    print(f"ID3 Model F1-score: ({f1}±{f1_diff})")
    print(f"Accuracy Model F1-score: ({acc}±{acc_diff})")
    print(f"SK Model F1-score: ({sk_f1}±{sk_f1_diff})")
    print(f"Accuracy Model F1-score: ({sk_acc}±{sk_acc_diff})")

    id3_tree = DecisionTree(criterion=criterion, max_depth=max_depth)
    id3_tree.fit(X_train, y_train)

    sk_tree = DecisionTreeClassifier(
        criterion=sk_criterion,
        max_depth=sk_max_depth,
        random_state=42,
        class_weight="balanced"
        )
    sk_tree.fit(X_train, y_train)

    return result, sk_result, id3_tree, sk_tree

def _get_permutation_importance(model, X, y, metric, n_repeats, seed) -> list:
    """
    Finds the importance score 

    Parameters
    ----------
    model: DecisionTree
        Trained decision tree model used for prediction.
    X: np.ndarray shape(n_samples, n_features)
        Input samples
    y: np.ndarray shape(n_samples)
        train data target values. Values must only include 0 or 1 
    metric: function
        Metric used to evaluate the model
    n_repeats: int
        Number of times each feature is permuted
    seed: int
        Seed used to split the data pseudo random

    Returns
    -------
    importance_scores: list 
        Permutation importance score for each feature
    """
    original_score = metric(y, model.predict(X))

    rng = np.random.default_rng(seed)  #tilfeldig generator med fast seed
    importance_scores = []
    
    num_features = X.shape[1] 
    for feature_i in range(num_features):
        permuted_scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy() #ødelegger ikke originaldata

            X_permuted[:, feature_i] = rng.permutation(X_permuted[:, feature_i])
        
            permuted_scores.append(metric(y, model.predict(X_permuted)))
            
        mean_score = np.mean(permuted_scores)

        importance = original_score - mean_score

        importance_scores.append(importance)

    return importance_scores
 
def visualize_data(X_train, y_train, feature_names) -> None:
    _, axes = plt.subplots(4, 5, figsize=(15, 10))
    n_features = X_train.shape[1]
 
    for i, ax in enumerate(axes.flat):
        if i < n_features:
            ax.hist(X_train[:, i], bins=15)
            ax.set_title(feature_names[i])
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        elif i == n_features:
            ax.hist(y_train, bins=2, rwidth=0.8)
            ax.set_title("Churn (Target)")
            ax.set_xlabel("Value (0 = Nei, 1 = Ja)")
            ax.set_ylabel("Count")
            ax.set_xticks([0, 1])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_permutation_importance(model, X_test, y_test):
    importance_scores = _get_permutation_importance(model, X_test, y_test, accuracy_score, 30, 42)

    sort = np.argsort(importance_scores)
    sorted_features = np.array(feature_names)[sort]
    sorted_scores = np.array(importance_scores)[sort]

    plt.barh(sorted_features, sorted_scores)
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("Feature importance")
    # plt.xticks(rotation="90")
    plt.tight_layout()
    plt.show()

def plot_f1_scores(result, sk_result):
    id3_criterion = result["best"]["criterion"]
    sk_criterion = sk_result["best"]["criterion"]

    id3_depths = [int(depth) for depth in result[id3_criterion].keys()]
    sk_depths = [int(depth) for depth in sk_result[sk_criterion].keys()]

    id3_score = [result[id3_criterion][depth]["val_f1"] for depth in id3_depths]
    sk_score = [sk_result[sk_criterion][depth]["val_f1"] for depth in sk_depths]

    plt.plot(id3_depths, id3_score, color="blue", label="ID3 model")
    plt.plot(sk_depths, sk_score, color="orange", label="Scikit model")

    plt.title("F1-scores on validation data for different depths")
    plt.xlabel("max_depth")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_test, id3_tree, sk_tree):
    y_pred = id3_tree.predict(X_test)
    sk_y_pred = sk_tree.predict(X_test)
    _, axes = plt.subplots(1, 2, figsize =(12, 5))

    # id3 tree
    cm_my = confusion_matrix(y_test, y_pred)
    disp_my = ConfusionMatrixDisplay(confusion_matrix=cm_my, display_labels=['No churn', 'Churn '])
    disp_my.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
    axes[0].set_title('Confusion Matrix: ID3 Decision Tree')
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    # Sklearn tree
    cm_sk = confusion_matrix(y_test, sk_y_pred)
    disp_sk = ConfusionMatrixDisplay(confusion_matrix=cm_sk, display_labels=['No churn', 'Churn '])
    disp_sk.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
    axes[1].set_title('Confusion Matrix: SK Decision Tree')
    axes[1].set_xlabel('Predicted label')
    axes[1].set_ylabel('True label')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from pathlib import Path

    X, y, feature_names = _load_data()
    X_train, X_test, y_train, y_test = _split_data(X,y)

    visualize_data(X_train, y_train, feature_names)

    model = DecisionTreeClassifier(max_depth=4, criterion="gini")
    model.fit(X_train, y_train)
    plot_permutation_importance(model, X_test, y_test)
    
    result, sk_result, id3_tree, sk_tree = run_pipeline()
    plot_f1_scores(result, sk_result)
    
    data = {
        "ID3_model":result,
        "Scikit_model": sk_result
        }
    content = json.dumps(data, indent=2)

    plot_permutation_importance(id3_tree, X_test, y_test)
    plot_confusion_matrix(y_test, id3_tree, sk_tree)

    Path("result.json").write_text(content, encoding="utf-8")


