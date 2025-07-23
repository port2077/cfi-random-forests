from collections import defaultdict
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score,roc_curve,auc,roc_auc_score


def get_cut_points(tree: sklearn.tree, z: Optional[List[int]] = None,x_j: Optional[int] = None ): # type: ignore

    '''tree is a single instance of the model of type sklearn.tree
    z is the list of feature indexes to be conditioned on
    x_j is the feature id on whose importance we need are trying to estimate
    returns features, thresholds which are the feature names and at what threshold the feature splits the data'''

    start_node = 0
    depth = 0
    root = (start_node,depth)
    tree_path = [root]
    features = np.zeros(tree.tree_.node_count)
    threshold = np.zeros(tree.tree_.node_count)

    while len(tree_path) != 0:
        node, depth = tree_path.pop(0)
        current_depth = depth + 1
        features[node] = tree.tree_.feature[node]
        threshold[node] = tree.tree_.threshold[node]

        if tree.tree_.children_right[node] != -1:
            tree_path.append((tree.tree_.children_right[node],current_depth)) # type: ignore

        if tree.tree_.children_left[node] != -1:
            tree_path.append((tree.tree_.children_left[node],current_depth)) # type: ignore

    # filter out the leaf nodes if z None and return all features with their thresholds
    # else return the chosen z features and their threshold
    if z :
        masks = np.isin(features,z)
        features = features[masks]
        threshold = threshold[masks]
    else:
        masks = ~np.isin(features,[x_j,-2]) # type: ignore
        features = features[masks]
        threshold = threshold[masks]
        

    return features,threshold

def prepare_data(df,features: np.array, thresholds: np.array, z: List[int], x_j: int): # type: ignore

    '''take the cutpoints and the features to be condtioned on
    and permute the dataset based on the condtional variables'''

    # cols = df.columns().tolist()
    grid_points = defaultdict(list)
    permuted_df = df.copy()

    for idx,row in df.iterrows():
        grid_loc = []
        for var in z:
            # get the feature thresholds by indexing the thresholds array with the feature index
            cutpoints = np.sort(thresholds[features == var])
            #breakpoint()
            #
            cut_at = np.searchsorted(cutpoints,row.iloc[var])
            grid_loc.append(cut_at)
        # this is a tuple of unique cutpoints and each such tuple contains a list of indices of the rows that fall in that cell
        grid_points[tuple(grid_loc)].append(idx)
    #breakpoint()
    # for each unique cutpoint tuple, permute all the values in the feature x_j for all the rows that fall in that tuple
    for cell, indices in grid_points.items():
            if len(indices) > 1:
                values = df.iloc[indices, x_j].values
                permuted_values = np.random.permutation(values)
                permuted_df.loc[permuted_df.index[indices], permuted_df.columns[x_j]] = permuted_values


    return permuted_df

def get_cfi_scores(model,test_df,target):

    '''get the conditional feature importance score for all features - Random Forest Models'''

    cfi_score_dict = defaultdict(int)
    n_estimators = model.n_estimators
    features = test_df.columns.tolist()
    # get the feature names and indices - this will be used to get x_j,z pairs where x_j is the feature 
    # whose importance we are looking for and z is the set of all features to be conditioned on
    feature_set = dict(zip(features, range(len(features))))
    test_df = test_df.reset_index(drop=True)
    
    # ietrate through all features 
    for feature in features:
        cfi_score = 0
        set_copy = feature_set.copy()
        x_j = set_copy.pop(feature)
        z = list(set_copy.values())

        for tree in model.estimators_:
            
            f,t = get_cut_points(tree,z,x_j)
            permuted_df = prepare_data(test_df,f,t,z,x_j)
            true_scores = tree.predict(test_df.to_numpy())
            true_accuracy = accuracy_score(target, true_scores)
            conditional_scores = tree.predict(permuted_df.to_numpy())
            condtional_accuracy = accuracy_score(target, conditional_scores)
            cfi_score += true_accuracy - condtional_accuracy
       # breakpoint()
        cfi_score_dict[feature] = cfi_score/n_estimators

    return cfi_score_dict

def get_dt_cfi_scores(model,test_df,target):

    '''get the conditional feature importance score for all features - Decision Tree Models'''

    cfi_score_dict = defaultdict(int)
    features = test_df.columns.tolist()
    feature_set = dict(zip(features, range(len(features))))
    test_df = test_df.reset_index(drop=True)
    
    for feature in features:
        cfi_score = 0
        set_copy = feature_set.copy()
        x_j = set_copy.pop(feature)
        z = list(set_copy.values())

        f,t = get_cut_points(model,z,x_j)
        permuted_df = prepare_data(test_df,f,t,z,x_j)
        true_scores = model.predict(test_df.to_numpy())
        true_accuracy = accuracy_score(target, true_scores)
        conditional_scores = model.predict(permuted_df.to_numpy())
        condtional_accuracy = accuracy_score(target, conditional_scores)
        cfi_score += true_accuracy - condtional_accuracy
       # breakpoint()
        cfi_score_dict[feature] = cfi_score

    return cfi_score_dict



if __name__ == '__main__':
    
    model = joblib.load('models/rf.joblib')
    df = pd.read_csv('data/test_df.csv',index_col=False)
    target = df['label'].copy()
    test_df = df.drop('label', axis=1)
    score = get_cfi_scores(model,test_df,target)
    print(f'cfi score {score}')
    