from collections import defaultdict
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score,roc_curve,auc,roc_auc_score

# def run():
#     model = joblib.load('models/rf.joblib')
#     print('estimator',model.estimator_)
#     print('estimators',model.estimators_)
#     print('classes',model.classes_)
#     print('n_classes',model.estimator_)


def get_cut_points(tree: sklearn.tree, z: Optional[List[int]] = None,x_j: Optional[int] = None ): # type: ignore

    '''tree is a single instance of the model of type sklearn.tree
    z is the list of feature indexes to be conditioned on
    x_j is the feature id on whose importance we need are trying to estimate'''

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
            cutpoints = np.sort(thresholds[features == var])
            #breakpoint()
            cut_at = np.searchsorted(cutpoints,row.iloc[var])
            grid_loc.append(cut_at)
        
        grid_points[tuple(grid_loc)].append(idx)
    #breakpoint()
    for cell, indices in grid_points.items():
            if len(indices) > 1:
                values = df.iloc[indices, x_j].values
                permuted_values = np.random.permutation(values)
                permuted_df.loc[permuted_df.index[indices], permuted_df.columns[x_j]] = permuted_values


    return permuted_df

def get_cfi_scores(model,test_df,target):

    cfi_score = 0
    n_estimators = model.n_estimators
    z = [5,7]

    for tree in model.estimators_:
        
        f,t = get_cut_points(tree,z,9)
        permuted_df = prepare_data(test_df,f,t,z,9)
        true_scores = tree.predict(test_df)
        conditional_scores = tree.predict(permuted_df)
        true_accuracy = accuracy_score(target, true_scores)
        condtional_accuracy = accuracy_score(target, conditional_scores)
        cfi_score += true_accuracy - condtional_accuracy
       # breakpoint()

    return cfi_score/n_estimators



if __name__ == '__main__':
    
    df = pd.read_csv('data/test_df.csv',index_col=False)
    target = df['admission_label'].copy()
    features = df.drop('admission_label', axis=1)
    model = joblib.load('models/rf.joblib')
    # z = [5,7]
    # f,t = get_cut_points(model.estimators_[0],z,9)
    # print(f,t)
    # # breakpoint()
    # gr = prepare_data(df,f,t,z,9)
    score = get_cfi_scores(model,features,target)
    print(f'cfi score {score}')
    