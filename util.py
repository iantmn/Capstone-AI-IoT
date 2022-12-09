# module for utility functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

# Function to extract how important features are based on the model (standard uses random forest)
def computeFeatureImportance(df_X: pd.DataFrame[float], df_Y: pd.DataFrame[float], n_repeats: int = 10, model: object = None, scoring: object = None) -> pd.DataFrame[float, str]:
    """Compute the feature importance of a model.

    Args:
        df_X (pd.DataFrame): Dataframe of the features.
        df_Y (pd.DataFrame): Dataframe of the labels.
        n_repeats (int, optional): Number of repeats. Defaults to 10.
        model (object, optional): Model to do the test on. Defaults to None.
        scoring (object, optional): How scoring is done. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of the feature importance. The higher the value, the more important the feature is. First column is the feature importance, second column is the feature name.
    """    
    # standard model is random forest
    if model is None:
        model = RandomForestClassifier(random_state=42)
    print(f"Computer feature importance using {model}...")
    # Fit the model
    model.fit(df_X, df_Y.squeeze())
    # Get the feature importance
    result = permutation_importance(model, df_X, df_Y,
                                    n_repeats=n_repeats, random_state=42, scoring=scoring)
    # Processing the result
    feat_names = df_X.columns.copy()
    feat_ims = np.array(result.importances_mean)
    sorted_ims_idx = np.argsort(feat_ims)[::-1]
    feat_names = feat_names[sorted_ims_idx]
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    df_res = pd.DataFrame()
    df_res["feature_importance"] = feat_ims
    df_res["feature_name"] = feat_names
    return df_res