import pandas as pd
from bank.pipelines import make_linear_preprocessor, make_tree_preprocessor

def test_preprocessors_fit_transform():
    X = pd.DataFrame({
        "num1":[1,2,3,None],
        "cat1":["a","b","a","c"]
    })
    num = ["num1"]; cat=["cat1"]
    pre_lin = make_linear_preprocessor(num, cat)
    pre_tree = make_tree_preprocessor(num, cat)
    Xt1 = pre_lin.fit_transform(X)
    Xt2 = pre_tree.fit_transform(X)
    assert Xt1.shape[0] == X.shape[0]
    assert Xt2.shape[0] == X.shape[0]
