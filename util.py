# module for utility functions

# Function to extract how important features are based on the model (standard uses random forest)
def computeFeatureImportance(df_X, df_Y, model=None, scoring=None):
    if model is None:
        model = RandomForestClassifier(random_state=0)
    print("Computer feature importance using", model)
    model.fit(df_X, df_Y.squeeze())
    result = permutation_importance(model, df_X, df_Y,
                                    n_repeats=10, random_state=0, scoring=scoring)
    feat_names = df_X.columns.copy()
    feat_ims = np.array(result.importances_mean)
    sorted_ims_idx = np.argsort(feat_ims)[::-1]
    feat_names = feat_names[sorted_ims_idx]
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    df = pd.DataFrame()
    df["feature_importance"] = feat_ims
    df["feature_name"] = feat_names
    return df