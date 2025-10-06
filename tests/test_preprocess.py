import pandas as pd
from preprocess import prepare_frames

def test_prepare_frames_basic(tmp_path):
    train = pd.DataFrame({
        "id":[1,2,3,4],
        "age":[30,40,35,50],
        "default":["no","yes","no","no"],
        "pdays":[-1,10,-1,5],
        "y":[0,1,0,1],
    })
    test = pd.DataFrame({
        "id":[5,6],
        "age":[28,60],
        "default":["no","yes"],
        "pdays":[-1,-1],
    })
    tp = tmp_path/"train.csv"; train.to_csv(tp, index=False)
    ep = tmp_path/"test.csv";  test.to_csv(ep, index=False)
    cfg = {
        "data":{"target":"y","drop_cols":["id"],"yn_binary_cols":["default"],"use_duration":False},
        "features":{"numeric":["age","duration","pdays","previous"],"categorical":["default"],"add_pdays_indicator":True}
    }
    X,y,Xt,num,cat = prepare_frames(tp, ep, cfg)
    assert "pdays_is_never" in X.columns
    assert set(num).issubset(set(X.columns))
    assert set(cat).issubset(set(X.columns))
