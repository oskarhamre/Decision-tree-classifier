from enum import Enum
import numpy as np
import pprint


class Criterion(Enum):
    """
    Purity measures

    Attributes
    ----------
    ENTROPY: str
        H(x) = -Sum( p_i * log2(p_i) )
    GINI: str
        G(x) = Sum( p_i * (1-p_i) )
    """
    ENTROPY = "entropy"
    GINI = "gini"


class DecisionTree:
    """A binary decision tree calssifier using the ID3 algorithm"""

    criterion: Criterion
    max_depth: int | None = None
    tree: dict
    depth: int = 0

    def __init__(self, criterion: str = "entropy" , max_depth: int | None = None) -> None: # int or None
        """
        Initialize the decision tree

        Parameters
        ------------
        criterion: str, default="entropy"
            How to measure purity in label.
            Supported criteria are "entropy" and "gini"
        max_depth: int or None, default=None
            Maximum depth of the tree.
            None will keep growing until all leafes are pure or no features to split on
        """
        self.criterion = Criterion(criterion)
        self.max_depth = max_depth
        self.tree = dict()

    def print_tree(self) -> None:
        """Prints the tree in a readable format"""
        pprint.pp(self.tree) # more readable format

    def fit(self, X: np.ndarray, y:np.ndarray) -> None:
            """
            Trains the model on data
    
            Parameters
            ----------
            X: np.ndarray shape(n_samples, n_features)
                train data input samples, cleaned without feature names
            y: np.ndarray shape(n_samples)
                train data target values. Values must only include 0 or 1        
            """
            self.tree = self._make_node(0, X, y)

    def predict(self, X) -> np.ndarray:
            """
            Predicts the outcome of given data
    
            Parameters
            ----------
            X: np.ndarray shape(n_samples, n_features)
                Input samples
    
            Returns
            -------
            y_pred: np.ndarray shape(n_samples)
                Predicted outcome of input samples
            """
            y_pred = []
    
            for x in X:
                node = self.tree
                while "feature_i" in node:
                    if x[node["feature_i"]] < node["threshold"]:
                    #if x["feature_i"] < node["threshold"]:
                        node = node["left"]
                    else:
                        node = node["right"]
                        
                y_pred.append(node["label"])
                
            return np.array(y_pred)

    def get_depth(self) -> int:
        """Returns actual depth of the tree, from root to leaf"""
        return self.depth
    
    def _get_children(self, X: np.ndarray, y: np.ndarray, i: int, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Splits the data X and y by the feature on index i and threshold thresh"""
        col = X[:, i]

        left_mask = col < thresh
        right_mask = ~left_mask

        X_left = X[left_mask]
        X_right = X[right_mask]
        y_left = y[left_mask]
        y_right = y[right_mask]

        return X_left, X_right, y_left, y_right, float(thresh)

    def _get_purity(self, y: np.ndarray) -> float:
        """Calculates purity on 1D array y using chosen criterion"""
        if y.size == 0: # prevents division by zero
            return 0.0
    
        p = np.sum(y)/y.size

        if p == 0.0 or p == 1.0:
            return 0.0

        elif self.criterion == Criterion.ENTROPY:
            # calculate entropy score
            return - ((p*np.log2(p)) + ((1-p)*np.log2(1-p))) 
        
        elif self.criterion == Criterion.GINI:
            # calculate gini score
            return 2*p*(1-p)
        
        else:
            # throws IllegalArgumentException if purity not defined in Enum
            raise ValueError("Unknown criterion")

    def _get_dependent_purity(self, X: np.ndarray, y: np.ndarray, i: int, thresh: float) -> float:
        """Splits the data using _get_children and calculates each purity. Returns the weighted mean of these"""
        _,_, y_left, y_right,_ = self._get_children(X, y, i, thresh)
        
        purity_left = self._get_purity(y_left)
        purity_right = self._get_purity(y_right)

        # returns weighted purity
        total_len = y.size
        left_len = y_left.size
        right_len = y_right.size
        weighted_purity = ((left_len / total_len) * purity_left + (right_len / total_len) * purity_right) 
        return weighted_purity

    def _get_information_gain(self, X: np.ndarray, y: np.ndarray, i: int, thresh: float) -> float:
        """returns IG(x) = H(y) - H(y|x) where H(y) is purity and H(y|x) is dependent purity"""
        purity = self._get_purity(y)
        dependent_purity = self._get_dependent_purity(X, y, i, thresh)
        information_gain = purity - dependent_purity

        return information_gain

    def _get_best_split(self, X: np.ndarray, y: np.ndarray) -> None | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Finds the feature split with highest information gain, returns None if no split usefull"""
        best_i = 0
        best_info_gain = None
        best_split = None

        num_features = X.shape[1]
        ratio = np.sum(y) / y.size
        p = ratio * 100

        for i in range(num_features):
            col = X[:, i]

            for percent in [25, p, 50, 75, 100-p]:
                thresh = np.percentile(col, percent)
                X_left, X_right, y_left, y_right, thresh = self._get_children(X, y, i, thresh)

                if len(y_right) == 0 or len(y_left) == 0:
                    continue

                info_gain = self._get_information_gain(X, y, i, thresh)

                if best_info_gain is None or info_gain > best_info_gain :
                    best_info_gain = info_gain
                    best_split = (X_left, X_right, y_left, y_right, thresh, i)

        if best_split is None or best_info_gain is None or best_info_gain <= 0:
            return None
        
        return best_split

    def _make_node(self, depth: int, X: np.ndarray, y: np.ndarray) -> dict:
        """Returns a dict containing a label or new node"""
        if depth>self.depth:
            self.depth = depth

        if len(np.unique(y)) == 1:
            # make leaf
            label = y[0]
            return {"label": label}
        
        elif np.all(X==X[0]) or (self.max_depth != None and depth >= self.max_depth):
            # make leaf
            label = np.bincount(y).argmax()
            return {"label": label}
        
        split = self._get_best_split(X, y)

        if split is None:
            label = np.bincount(y).argmax()
            return {"label": label}            

        X_left, X_right, y_left, y_right, thresh, feature_i = split


        tree = {
            "feature_i": feature_i,
            "threshold": thresh,
            "left": self._make_node(depth+1, X_left, y_left),
            "right": self._make_node(depth+1, X_right, y_right)
        }
        return tree
    