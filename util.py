# module for utility functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

# Function to extract how important features are based on the model (standard uses random forest)
def computeFeatureImportance(df_X, df_Y, model=None, scoring=None):
    """Compute the feature importance of a model.

    Args:
        df_X (_type_): Dataframe of the features.
        df_Y (_type_): Dataframe of the labels.
        model (_type_, optional): Model to do the test on. Defaults to None.
        scoring (_type_, optional): How scoring is done. Defaults to None.

    Returns:
        _type_: _description_
    """    
    # standard model is random forest
    if model is None:
        model = RandomForestClassifier(random_state=42)
    print("Computer feature importance using", model)
    # Fit the model
    model.fit(df_X, df_Y.squeeze())
    # Get the feature importance
    result = permutation_importance(model, df_X, df_Y,
                                    n_repeats=10, random_state=42, scoring=scoring)
    # Processing the result
    feat_names = df_X.columns.copy()
    feat_ims = np.array(result.importances_mean)
    sorted_ims_idx = np.argsort(feat_ims)[::-1]
    feat_names = feat_names[sorted_ims_idx]
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    df = pd.DataFrame()
    df["feature_importance"] = feat_ims
    df["feature_name"] = feat_names
    return df