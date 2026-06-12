import json
from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from shap import Explainer


LABEL_NAMES = ['loss1_mean']

def _get_shap_values(model, explainer_cls: type[Explainer],
                     xs_train: np.ndarray, xs_test: np.ndarray
                     ) -> np.ndarray:
    explainer = explainer_cls(model, xs_train)
    # shap_values = explainer.shap_values(xs_test)  # shap值
    # print(shap_values.shape)
    # shap.summary_plot(shap_values, xs_test, show=False)
    shap_results = explainer(xs_test)
    shap_values = shap_results.values
    shap_features = shap_results.data
    return shap_values, shap_features

def train_logistic_regression(n_story: int, X: np.ndarray, y: np.ndarray,
                              ax: Axes, seed: int | None, gs: bool=False,
                              shap_ana: bool=False):
    """训练逻辑回归模型"""
    model_name = 'LR'
    print(f'=========================== Logistic Regression ===========================')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    Xs_train = x_scaler.fit_transform(X_train)
    Xs_test = x_scaler.transform(X_test)

    if n_story == 4:
        paras = {'C': 0.1, 'solver': 'newton-cg', 'l1_ratio': 0}
    elif n_story == 8:
        paras = {'C': 10, 'solver': 'newton-cg', 'l1_ratio': 0}
    elif n_story == 12:
        paras = {'C': 0.01, 'solver': 'liblinear', 'l1_ratio': 0}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = LogisticRegression(
        random_state=seed,
        **paras,
    )

    # 带交叉验证的网格搜索
    if gs:
        C = [0.01, 0.1, 1, 10]
        solver = ['newton-cg', 'lbfgs', 'liblinear','sag','saga']
        l1_ratio = [0, 1]
        param_grid = {
            'C': C,
            'solver': solver,
            'l1_ratio': l1_ratio,
        }
        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=1)
        grid_search.fit(Xs_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    y_pred_train = model.predict(Xs_train)
    y_prob = model.predict_proba(Xs_test)[:, 1]
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(classification_report(y_test, y_pred, target_names=['0', '1']))
    print(f'混淆矩阵:\n{cm_test}')
    x_corr, y_corr, x_incorr, y_incorr = _plot_scatter(ax, y_test, y_pred, y_prob, model_name, acc_test)
    print(f'Accuracy (train): {acc_train:.4f}')
    print(f'Accuracy (test): {acc_test:.4f}')
    output = {
        'model': model,
        'x_scaler': x_scaler
    }
    joblib.dump(output, f'models/classifiers/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, X_test, X_train, np.column_stack((y_test, y_pred)), np.column_stack((y_train, y_pred_train)))
    _output_result(n_story, model_name, acc_train, acc_test, cm_test, cm_train,
                   x_corr, y_corr, x_incorr, y_incorr)
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, shap.LinearExplainer, Xs_train, Xs_test)
        np.savetxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Classification/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_xgboost(n_story: int, X: np.ndarray, y: np.ndarray, ax: Axes,
                  seed: int | None, gs: bool=False, shap_ana: bool=False):
    """训练xgboost模型"""
    model_name = 'XGBoost'
    print(f'=========================== XGBoost ===========================')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    Xs_train = x_scaler.fit_transform(X_train)
    Xs_test = x_scaler.transform(X_test)

    if n_story == 4:
        paras = {'gamma': 0.5, 'max_depth': 3, 'min_child_weight': 1, 'colsample_bytree': 1, 'learning_rate': 0.1, 'n_estimators': 300,'subsample': 0.7}
    elif n_story == 8:
        paras = {'gamma': 0, 'max_depth': 3, 'min_child_weight': 1, 'colsample_bytree': 1, 'learning_rate': 0.3, 'n_estimators': 500,'subsample': 0.7}
    elif n_story == 12:
        paras = {'gamma': 0, 'max_depth': 12, 'min_child_weight': 1, 'colsample_bytree': 1, 'learning_rate': 0.3, 'n_estimators': 300,'subsample': 0.7}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = XGBClassifier(
        random_state=seed,
        **paras
    )

    # 带交叉验证的网格搜索
    if gs:
        max_depth = [3, 5, 6, 7, 9, 12, 15, 17, 25]
        min_child_weight = [1, 3, 5, 7]
        gamma = [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        subsample = [0.6, 0.7, 0.8, 0.9, 1]
        colsample_bytree = [0.6, 0.7, 0.8, 0.9, 1]
        n_estimators = [100, 200, 300, 400, 500]
        learning_rate = [0.01, 0.015, 0.025, 0.05, 0.1, 0.3]
        param_grid = {
            # 'max_depth': max_depth,
            # 'min_child_weight': min_child_weight,
            # 'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
        }
        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=-1)
        grid_search.fit(Xs_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    y_pred_train = model.predict(Xs_train)
    y_prob = model.predict_proba(Xs_test)[:, 1]
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(classification_report(y_test, y_pred, target_names=['0', '1']))
    print(f'混淆矩阵:\n{cm_test}')
    x_corr, y_corr, x_incorr, y_incorr = _plot_scatter(ax, y_test, y_pred, y_prob, model_name, acc_test)
    print(f'Accuracy (train): {acc_train:.4f}')
    print(f'Accuracy (test): {acc_test:.4f}')
    output = {
        'model': model,
        'x_scaler': x_scaler
    }
    joblib.dump(output, f'models/classifiers/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, X_test, X_train, np.column_stack((y_test, y_pred)), np.column_stack((y_train, y_pred_train)))
    _output_result(n_story, model_name, acc_train, acc_test, cm_test, cm_train,
                   x_corr, y_corr, x_incorr, y_incorr)
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, shap.TreeExplainer, Xs_train, Xs_test)
        np.savetxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Classification/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_random_forest(n_story: int, X: np.ndarray, y: np.ndarray, ax: Axes,
                        seed: int | None, gs: bool=False, shap_ana: bool=False):
    """训练随机森林模型"""
    model_name = 'RF'
    print(f'=========================== Random Forest ===========================')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    Xs_train = x_scaler.fit_transform(X_train)
    Xs_test = x_scaler.transform(X_test)

    if n_story == 4:
        paras = {'n_estimators': 200, 'max_depth': 12}
    elif n_story == 8:
        paras = {'n_estimators': 200, 'max_depth': 15}
    elif n_story == 12:
        paras = {'n_estimators': 500, 'max_depth': 15}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = RandomForestClassifier(
        criterion='gini',
        random_state=seed,
        **paras
    )

    # 带交叉验证的网格搜索
    if gs:
        n_estimators = [100, 200, 300, 400, 500]
        max_depth = [3, 5, 6, 7, 9, 12, 15, 17, 25]
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=1)
        grid_search.fit(Xs_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    y_pred_train = model.predict(Xs_train)
    y_prob = model.predict_proba(Xs_test)[:, 1]
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(classification_report(y_test, y_pred, target_names=['0', '1']))
    print(f'混淆矩阵:\n{cm_test}')
    x_corr, y_corr, x_incorr, y_incorr = _plot_scatter(ax, y_test, y_pred, y_prob, model_name, acc_test)
    print(f'Accuracy (train): {acc_train:.4f}')
    print(f'Accuracy (test): {acc_test:.4f}')
    output = {
        'model': model,
        'x_scaler': x_scaler
    }
    joblib.dump(output, f'models/classifiers/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, X_test, X_train, np.column_stack((y_test, y_pred)), np.column_stack((y_train, y_pred_train)))
    _output_result(n_story, model_name, acc_train, acc_test, cm_test, cm_train,
                   x_corr, y_corr, x_incorr, y_incorr)
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, shap.TreeExplainer, Xs_train, Xs_test)
        np.savetxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Classification/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_ann(n_story: int, X: np.ndarray, y: np.ndarray, ax: Axes,
              seed: int | None, gs: bool=False, shap_ana: bool=False):
    """训练人工神经网络模型"""
    model_name = 'ANN'
    print(f'=========================== ANN ===========================')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    Xs_train = x_scaler.fit_transform(X_train)
    Xs_test = x_scaler.transform(X_test)

    if n_story == 4:
        paras = {'hidden_layer_sizes': (16, 8), 'learning_rate_init': 0.01}
    elif n_story == 8:
        paras = {'hidden_layer_sizes': (128, 64), 'learning_rate_init': 0.01}
    elif n_story == 12:
        paras = {'hidden_layer_sizes': (64, 32), 'learning_rate_init': 0.01}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = MLPClassifier(
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
        **paras
        
    )

    # 带交叉验证的网格搜索
    if gs:
        hidden_layer_sizes = [(8,), (16), (32), (16, 8), (32, 16), (64, 32), (128, 64)]
        learning_rate_init = [0.001, 0.01, 0.1, 1]
        param_grid = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'learning_rate_init': learning_rate_init,
        }
        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=16)
        grid_search.fit(Xs_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    y_pred_train = model.predict(Xs_train)
    y_prob = model.predict_proba(Xs_test)[:, 1]
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(classification_report(y_test, y_pred, target_names=['0', '1']))
    print(f'混淆矩阵:\n{cm_test}')
    x_corr, y_corr, x_incorr, y_incorr = _plot_scatter(ax, y_test, y_pred, y_prob, model_name, acc_test)
    print(f'Accuracy (train): {acc_train:.4f}')
    print(f'Accuracy (test): {acc_test:.4f}')
    output = {
        'model': model,
        'x_scaler': x_scaler
    }
    joblib.dump(output, f'models/classifiers/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, X_test, X_train, np.column_stack((y_test, y_pred)), np.column_stack((y_train, y_pred_train)))
    _output_result(n_story, model_name, acc_train, acc_test, cm_test, cm_train,
                   x_corr, y_corr, x_incorr, y_incorr)
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, shap.DeepExplainer, Xs_train, Xs_test)
        np.savetxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Classification/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_svm(n_story: int, X: np.ndarray, y: np.ndarray, ax: Axes,
              seed: int | None, gs: bool=False, shap_ana: bool=False):
    """训练支持向量机模型"""
    model_name = 'SVM'
    print(f'=========================== SVM ===========================')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    Xs_train = x_scaler.fit_transform(X_train)
    Xs_test = x_scaler.transform(X_test)

    if n_story == 4:
        paras = {'C': 0.1, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    elif n_story == 8:
        paras = {'C': 10, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    elif n_story == 12:
        paras = {'C': 10, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = SVC(
        coef0=0.0,
        shrinking=True,
        probability=True,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',
        random_state=seed,
        **paras
    )

    # 带交叉验证的网格搜索
    if gs:
        C = [0.1, 1, 10]
        gamma = [0.001, 0.01, 0.1]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        degree = [2, 3, 4]
        param_grid = {
            'C': C,
            'gamma': gamma,
            'kernel': kernel,
            'degree': degree,
        }
        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=-1)
        grid_search.fit(Xs_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    y_pred_train = model.predict(Xs_train)
    y_prob = model.predict_proba(Xs_test)[:, 1]
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(classification_report(y_test, y_pred, target_names=['0', '1']))
    print(f'混淆矩阵:\n{cm_test}')
    x_corr, y_corr, x_incorr, y_incorr = _plot_scatter(ax, y_test, y_pred, y_prob, model_name, acc_test)
    print(f'Accuracy (train): {acc_train:.4f}')
    print(f'Accuracy (test): {acc_test:.4f}')
    output = {
        'model': model,
        'x_scaler': x_scaler
    }
    joblib.dump(output, f'models/classifiers/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, X_test, X_train, np.column_stack((y_test, y_pred)), np.column_stack((y_train, y_pred_train)))
    _output_result(n_story, model_name, acc_train, acc_test, cm_test, cm_train,
                   x_corr, y_corr, x_incorr, y_incorr)
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, shap.KernelExplainer, Xs_train, Xs_test)
        np.savetxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Classification/shap_features_{model_name}_{n_story}S.txt', shap_features)
    
def train_knn(n_story: int, X: np.ndarray, y: np.ndarray, ax: Axes,
              seed: int | None, gs: bool=False, shap_ana: bool=False):
    """训练K近邻模型"""
    model_name = 'KNN'
    print(f'=========================== KNN ===========================')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    Xs_train = x_scaler.fit_transform(X_train)
    Xs_test = x_scaler.transform(X_test)

    if n_story == 4:
        paras = {'leaf_size': 1, 'n_neighbors': 50}
    elif n_story == 8:
        paras = {'leaf_size': 1, 'n_neighbors': 15}
    elif n_story == 12:
        paras = {'leaf_size': 1, 'n_neighbors': 15}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = KNeighborsClassifier(
        weights='uniform',
        algorithm='auto',
        p=2,
        metric='minkowski',
        metric_params=None,
        **paras
    )

    # 带交叉验证的网格搜索
    if gs:
        n_neighbors = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        leaf_size = [1, 2, 3, 4, 5]
        param_grid = {
            'n_neighbors': n_neighbors,
            'leaf_size': leaf_size,
        }
        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=1)
        grid_search.fit(Xs_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    y_pred_train = model.predict(Xs_train)
    y_prob = model.predict_proba(Xs_test)[:, 1]
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(classification_report(y_test, y_pred, target_names=['0', '1']))
    print(f'混淆矩阵:\n{cm_test}')
    x_corr, y_corr, x_incorr, y_incorr = _plot_scatter(ax, y_test, y_pred, y_prob, model_name, acc_test)
    print(f'Accuracy (train): {acc_train:.4f}')
    print(f'Accuracy (test): {acc_test:.4f}')
    output = {
        'model': model,
        'x_scaler': x_scaler
    }
    joblib.dump(output, f'models/classifiers/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, X_test, X_train, np.column_stack((y_test, y_pred)), np.column_stack((y_train, y_pred_train)))
    _output_result(n_story, model_name, acc_train, acc_test, cm_test, cm_train,
                   x_corr, y_corr, x_incorr, y_incorr)
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, shap.KernelExplainer, Xs_train, Xs_test)
        np.savetxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Classification/shap_features_{model_name}_{n_story}S.txt', shap_features)
    
def _plot_scatter(
    ax: Axes,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    accuracy: float):
    """拟合结果可视化

    Args:
        y_test (np.ndarray): 测试集
        y_pred (np.ndarray): 预测结果
        y_prob (np.ndarray): 预测概率
        title (str): 标题
        accuracy (float): 准确率
    """
    jitter_x = y_test + np.random.uniform(-0.4, 0.4, size=len(y_test))  # 随机扰动
    correct_mask = (y_test == y_pred)
    incorrect_mask = (y_test != y_pred)
    x_corr, y_corr = jitter_x[correct_mask], y_prob[correct_mask]
    x_incorr, y_incorr = jitter_x[incorrect_mask], y_prob[incorrect_mask]

    ax.set_title(f'{title}: {accuracy:.4f}')
    ax.scatter(x_corr, y_corr, c='blue', alpha=0.6, label='Correct', edgecolors='k', s=50)  # 正确点
    ax.scatter(x_incorr, y_incorr, c='red', alpha=0.7, marker='x', label='Incorrect', s=80)  # 错误点
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1,
                label='0.5')  # 阈值线
    ax.set_ylabel('Probability')
    ax.set_xticks([0, 1], ['0', '1'])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    return x_corr, y_corr, x_incorr, y_incorr

def _output_result(n_story: int, model_name: str, acc_train: float, acc_test: float,
                   cm_test: np.ndarray, cm_train: np.ndarray,
                   x_corr: np.ndarray, y_corr: np.ndarray,
                   x_incorr: np.ndarray, y_incorr: np.ndarray):
    """输出结果"""
    data = {}
    data['model_name'] = model_name
    data['acc_train'] = acc_train
    data['acc_test'] = acc_test
    data['confusion_matrix_test'] = cm_test.tolist()
    data['confusion_matrix_train'] = cm_train.tolist()
    np.savetxt(f'ml_output/Classification/{model_name}_{n_story}S_correct.txt', np.column_stack((x_corr, y_corr)), fmt='%.6f')
    np.savetxt(f'ml_output/Classification/{model_name}_{n_story}S_incorrect.txt', np.column_stack((x_incorr, y_incorr)), fmt='%.6f')
    json.dump(data, open(f'ml_output/Classification/{model_name}_{n_story}S.json', 'w'), indent=4)

def _output(
    n_story: int,
    model_name: str,
    x_test: np.ndarray,
    x_train: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray):
    """输出结果"""
    if n_story == 4:
        feature_names = ['R', 'lamda', 'a1']
    elif n_story == 8:
        feature_names = ['R', 'lamda', 'a1', 'a2']
    elif n_story == 12:
        feature_names = ['R', 'lamda', 'a1', 'a2', 'a3']
    columns = ['Actual_' + name for name in LABEL_NAMES] + ['Predicted_' + name for name in LABEL_NAMES]
    df_x_test = pd.DataFrame(x_test, columns=feature_names)
    df_x_train = pd.DataFrame(x_train, columns=feature_names)
    df_y_test = pd.DataFrame(y_test, columns=columns)
    df_y_train = pd.DataFrame(y_train, columns=columns)
    df_x_test.to_csv(f'ml_output/Classification/{model_name}_{n_story}S_x_test.csv')
    df_x_train.to_csv(f'ml_output/Classification/{model_name}_{n_story}S_x_train.csv')
    df_y_test.to_csv(f'ml_output/Classification/{model_name}_{n_story}S_y_test.csv')
    df_y_train.to_csv(f'ml_output/Classification/{model_name}_{n_story}S_y_train.csv')

def plot_shap(n_story: int, feature_names: list[str], model_name: str, label_to_plot: str):
    """绘制shap图"""
    df_x = pd.read_csv(f'ml_output/Classification/{model_name}_{n_story}S_x_test.csv', index_col=0)
    df_y = pd.read_csv(f'ml_output/Classification/{model_name}_{n_story}S_y_test.csv', index_col=0)
    x_test = df_x.to_numpy()
    y_test = df_y[f'Actual_{label_to_plot}'].to_numpy()
    shap_values: np.ndarray = np.loadtxt(f'ml_output/Classification/shap_values_{model_name}_{n_story}S.txt')
    if len(LABEL_NAMES) > 1:
        idx = LABEL_NAMES.index(label_to_plot)
        shap_values = shap_values[:, :, idx]
    plt.title(f'{model_name} (Feature={label_to_plot})')
    shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
    plt.savefig(f'ml_output/Classification/shap_{model_name}_{n_story}S_{label_to_plot}.png', dpi=600)
    plt.show()
    # 计算并可视化特征重要性
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance_df.to_csv(f'ml_output/Classification/{model_name}_{n_story}S_{label_to_plot}_importance.csv')
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.title(f'{model_name} (Feature={label_to_plot})')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'ml_output/Classification/{model_name}_{n_story}S_{label_to_plot}_importance.png', dpi=600)
    plt.show()

if __name__ == "__main__":
    seed = 42
    for n_story in [4, 8, 12]:
        data1 = pd.read_csv(f"data/data1_{n_story}S.csv", index_col=0)
        if n_story == 4:
            cols_a = ['a1']
        elif n_story == 8:
            cols_a = ['a1', 'a2']
        elif n_story == 12:
            cols_a = ['a1', 'a2', 'a3']
        feature_names = ['R', 'lamda'] + cols_a
        x = data1[['R', 'lamda', *cols_a]].to_numpy()
        y = data1['ok'].to_numpy(dtype=int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        gs = False
        Path('ml_output/Classification').mkdir(parents=True, exist_ok=True)
        Path('models/classifiers').mkdir(parents=True, exist_ok=True)
        train_logistic_regression(n_story, x, y, axes[0][0], seed, gs)
        train_knn(n_story, x, y, axes[0][1], seed, gs)
        train_random_forest(n_story, x, y, axes[0][2], seed, gs)
        train_xgboost(n_story, x, y, axes[1][0], seed, gs, shap_ana=True)
        train_svm(n_story, x, y, axes[1][1], seed, gs)
        train_ann(n_story, x, y, axes[1][2], seed, gs)
        plt.tight_layout()
        plt.savefig(f'ml_output/Classification/Results_{n_story}S.png', dpi=600)
        plt.show()
        plot_shap(n_story, feature_names, 'XGBoost', label_to_plot='loss1_mean')
