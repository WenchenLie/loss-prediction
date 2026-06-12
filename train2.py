from pathlib import Path
import joblib
import shap
from shap import Explainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


LABEL_NAMES = ['loss1_mean']
VERBOSE = 1
FEATURE_NAMES = {
    '4': ['R', 'lamda', 'a1'],
    '8': ['R', 'lamda', 'a1', 'a2'],
    '12': ['R', 'lamda', 'a1', 'a2', 'a3']
}

def _get_shap_explainer(model_name: str) -> type[Explainer]:
    if model_name == 'LR':
        return shap.LinearExplainer
    elif model_name in ['RF', 'XGBoost']:
        return shap.TreeExplainer
    else:
        return shap.KernelExplainer

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

def train_linear_regression(n_story: int, x: np.ndarray, y: np.ndarray, seed: int | None, shap_ana: bool = False, gs: bool=False):
    """训练线性回归模型"""
    model_name = 'LR'
    print(f'=========================== Linear Regression ===========================')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.fit_transform(y_train)
    xs_test = x_scaler.transform(x_test)

    model = LinearRegression()

    model.fit(xs_train, ys_train)
    ys_pred_test = model.predict(xs_test).reshape(-1, len(LABEL_NAMES))
    ys_pred_train = model.predict(xs_train).reshape(-1, len(LABEL_NAMES))
    y_pred_test = y_scaler.inverse_transform(ys_pred_test)
    y_pred_train = y_scaler.inverse_transform(ys_pred_train)
    y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
    y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
    y_pred_test = np.where(y_pred_test > 1, 1, y_pred_test)
    y_pred_train = np.where(y_pred_train > 1, 1, y_pred_train)
    res = pd.DataFrame(columns=LABEL_NAMES+['global'],
                       index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
    for i in range(len(LABEL_NAMES)):
        label = LABEL_NAMES[i]  # 特征名
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        rmse_test = root_mean_squared_error(y_test[:, i], y_pred_test[:, i])
        rmse_train = root_mean_squared_error(y_train[:, i], y_pred_train[:, i])
        res.loc['R2_test', label] = r2_test
        res.loc['R2_train', label] = r2_train
        res.loc['RMSE_test', label] = rmse_test
        res.loc['RMSE_train', label] = rmse_train
    global_R2_test = r2_score(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_R2_train = r2_score(y_train.reshape(-1), y_pred_train.reshape(-1))
    global_RMSE_test = root_mean_squared_error(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_RMSE_train = root_mean_squared_error(y_train.reshape(-1), y_pred_train.reshape(-1))
    res.loc['R2_test', 'global'] = global_R2_test
    res.loc['R2_train', 'global'] = global_R2_train
    res.loc['RMSE_test', 'global'] = global_RMSE_test
    res.loc['RMSE_train', 'global'] = global_RMSE_train
    # _plot_scatter(ax, y_test, y_pred_test, model_name, res.loc['R2_test', label_to_plot])
    print(res)
    output = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }
    joblib.dump(output, f'models/regressors/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, res, x_test, x_train, np.column_stack((y_test, y_pred_test)), np.column_stack((y_train, y_pred_train)))
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, _get_shap_explainer(model_name), xs_train, xs_test)
        np.savetxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Regression/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_knn(n_story: int, x: np.ndarray, y: np.ndarray, seed: int | None, shap_ana: bool = False, gs: bool=False):
    """训练K近邻模型"""
    model_name = 'KNN'
    print(f'=========================== KNN ===========================')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.fit_transform(y_train)
    xs_test = x_scaler.transform(x_test)

    if n_story == 4:
        paras = {'n_neighbors': 3, 'leaf_size': 1}
    elif n_story == 8:
        paras = {'n_neighbors': 5, 'leaf_size': 1}
    elif n_story == 12:
        paras = {'n_neighbors': 5, 'leaf_size': 1}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = KNeighborsRegressor(
        weights='uniform',
        algorithm='auto',
        p=2,
        metric='minkowski',
        metric_params=None,
        **paras
    )  # 网格搜索结果: n_neighbors=6, leaf_size=1

    # 带交叉验证的网格搜索
    if gs:
        n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        leaf_size = [1, 2]
        param_grid = {
            'n_neighbors': n_neighbors,
            'leaf_size': leaf_size,
        }
        scorer = make_scorer(r2_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=VERBOSE, n_jobs=16)
        grid_search.fit(xs_train, ys_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(xs_train, ys_train)
    ys_pred_test = model.predict(xs_test).reshape(-1, len(LABEL_NAMES))
    ys_pred_train = model.predict(xs_train).reshape(-1, len(LABEL_NAMES))
    y_pred_test = y_scaler.inverse_transform(ys_pred_test)
    y_pred_train = y_scaler.inverse_transform(ys_pred_train)
    y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
    y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
    y_pred_test = np.where(y_pred_test > 1, 1, y_pred_test)
    y_pred_train = np.where(y_pred_train > 1, 1, y_pred_train)
    res = pd.DataFrame(columns=LABEL_NAMES+['global'],
                       index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
    for i in range(len(LABEL_NAMES)):
        label = LABEL_NAMES[i]  # 特征名
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        rmse_test = root_mean_squared_error(y_test[:, i], y_pred_test[:, i])
        rmse_train = root_mean_squared_error(y_train[:, i], y_pred_train[:, i])
        res.loc['R2_test', label] = r2_test
        res.loc['R2_train', label] = r2_train
        res.loc['RMSE_test', label] = rmse_test
        res.loc['RMSE_train', label] = rmse_train
    global_R2_test = r2_score(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_R2_train = r2_score(y_train.reshape(-1), y_pred_train.reshape(-1))
    global_RMSE_test = root_mean_squared_error(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_RMSE_train = root_mean_squared_error(y_train.reshape(-1), y_pred_train.reshape(-1))
    res.loc['R2_test', 'global'] = global_R2_test
    res.loc['R2_train', 'global'] = global_R2_train
    res.loc['RMSE_test', 'global'] = global_RMSE_test
    res.loc['RMSE_train', 'global'] = global_RMSE_train
    # _plot_scatter(ax, y_test, y_pred_test, model_name, res.loc['R2_test', label_to_plot])
    print(res)
    output = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }
    joblib.dump(output, f'models/regressors/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, res, x_test, x_train, np.column_stack((y_test, y_pred_test)), np.column_stack((y_train, y_pred_train)))
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model.predict, _get_shap_explainer(model_name), xs_train, xs_test)
        np.savetxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Regression/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_random_forest(n_story: int, x: np.ndarray, y: np.ndarray, seed: int | None, shap_ana: bool = False, gs: bool=False):
    """训练随机森林模型"""
    model_name = 'RF'
    print(f'=========================== Random Forest ===========================')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.fit_transform(y_train)
    xs_test = x_scaler.transform(x_test)

    if n_story == 4:
        paras = {'n_estimators': 300, 'max_depth': 25}
    elif n_story == 8:
        paras = {'n_estimators': 100, 'max_depth': 25}
    elif n_story == 12:
        paras = {'n_estimators': 500, 'max_depth': 25}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = RandomForestRegressor(
        random_state=seed,
        **paras
    )  # 网格搜索结果: n_estimators=400, max_depth=20

    # 带交叉验证的网格搜索
    if gs:
        n_estimators = [100, 200, 300, 400, 500]
        max_depth = [15, 20, 25, 30, 35, 40]
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        scorer = make_scorer(r2_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=VERBOSE, n_jobs=16)
        grid_search.fit(xs_train, ys_train[:, 0])
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(xs_train, ys_train[:, 0])
    ys_pred_test = model.predict(xs_test).reshape(-1, len(LABEL_NAMES))
    ys_pred_train = model.predict(xs_train).reshape(-1, len(LABEL_NAMES))
    y_pred_test = y_scaler.inverse_transform(ys_pred_test)
    y_pred_train = y_scaler.inverse_transform(ys_pred_train)
    y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
    y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
    y_pred_test = np.where(y_pred_test > 1, 1, y_pred_test)
    y_pred_train = np.where(y_pred_train > 1, 1, y_pred_train)
    res = pd.DataFrame(columns=LABEL_NAMES+['global'],
                       index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
    for i in range(len(LABEL_NAMES)):
        label = LABEL_NAMES[i]  # 特征名
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        rmse_test = root_mean_squared_error(y_test[:, i], y_pred_test[:, i])
        rmse_train = root_mean_squared_error(y_train[:, i], y_pred_train[:, i])
        res.loc['R2_test', label] = r2_test
        res.loc['R2_train', label] = r2_train
        res.loc['RMSE_test', label] = rmse_test
        res.loc['RMSE_train', label] = rmse_train
    global_R2_test = r2_score(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_R2_train = r2_score(y_train.reshape(-1), y_pred_train.reshape(-1))
    global_RMSE_test = root_mean_squared_error(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_RMSE_train = root_mean_squared_error(y_train.reshape(-1), y_pred_train.reshape(-1))
    res.loc['R2_test', 'global'] = global_R2_test
    res.loc['R2_train', 'global'] = global_R2_train
    res.loc['RMSE_test', 'global'] = global_RMSE_test
    res.loc['RMSE_train', 'global'] = global_RMSE_train
    # _plot_scatter(ax, y_test, y_pred_test, model_name, res.loc['R2_test', label_to_plot])
    print(res)
    output = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }
    joblib.dump(output, f'models/regressors/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, res, x_test, x_train, np.column_stack((y_test, y_pred_test)), np.column_stack((y_train, y_pred_train)))
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, _get_shap_explainer(model_name), xs_train, xs_test)
        np.savetxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Regression/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_xgboost(n_story: int, x: np.ndarray, y: np.ndarray, seed: int | None, shap_ana: bool = False, gs: bool=False):
    """训练XGBoost模型"""
    model_name = 'XGBoost'
    print(f'=========================== XGBoost ===========================')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.fit_transform(y_train)
    xs_test = x_scaler.transform(x_test)

    if n_story == 4:
        paras = {'gamma': 0, 'max_depth': 25, 'min_child_weight': 7, 'colsample_bytree': 1, 'learning_rate': 0.05, 'n_estimators': 500,'subsample': 0.6}
    elif n_story == 8:
        paras = {'gamma': 0, 'max_depth': 6, 'min_child_weight': 7, 'colsample_bytree': 1, 'learning_rate': 0.05, 'n_estimators': 500,'subsample': 0.6}
    elif n_story == 12:
        paras = {'gamma': 0, 'max_depth': 25, 'min_child_weight': 7, 'colsample_bytree': 1, 'learning_rate': 0.025, 'n_estimators': 500,'subsample': 0.6}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = XGBRegressor(
        random_state= seed,
        **paras
    )  # 网格搜索结果: n_estimators=400, max_depth=20
    
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
        scorer = make_scorer(r2_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=VERBOSE, n_jobs=20)
        grid_search.fit(xs_train, ys_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(xs_train, ys_train)
    ys_pred_test = model.predict(xs_test).reshape(-1, len(LABEL_NAMES))
    ys_pred_train = model.predict(xs_train).reshape(-1, len(LABEL_NAMES))
    y_pred_test = y_scaler.inverse_transform(ys_pred_test)
    y_pred_train = y_scaler.inverse_transform(ys_pred_train)
    y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
    y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
    y_pred_test = np.where(y_pred_test > 1, 1, y_pred_test)
    y_pred_train = np.where(y_pred_train > 1, 1, y_pred_train)
    res = pd.DataFrame(columns=LABEL_NAMES+['global'],
                       index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
    for i in range(len(LABEL_NAMES)):
        label = LABEL_NAMES[i]  # 特征名
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        rmse_test = root_mean_squared_error(y_test[:, i], y_pred_test[:, i])
        rmse_train = root_mean_squared_error(y_train[:, i], y_pred_train[:, i])
        res.loc['R2_test', label] = r2_test
        res.loc['R2_train', label] = r2_train
        res.loc['RMSE_test', label] = rmse_test
        res.loc['RMSE_train', label] = rmse_train
    global_R2_test = r2_score(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_R2_train = r2_score(y_train.reshape(-1), y_pred_train.reshape(-1))
    global_RMSE_test = root_mean_squared_error(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_RMSE_train = root_mean_squared_error(y_train.reshape(-1), y_pred_train.reshape(-1))
    res.loc['R2_test', 'global'] = global_R2_test
    res.loc['R2_train', 'global'] = global_R2_train
    res.loc['RMSE_test', 'global'] = global_RMSE_test
    res.loc['RMSE_train', 'global'] = global_RMSE_train
    # _plot_scatter(ax, y_test, y_pred_test, model_name, res.loc['R2_test', label_to_plot])
    print(res)
    output = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }
    joblib.dump(output, f'models/regressors/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, res, x_test, x_train, np.column_stack((y_test, y_pred_test)), np.column_stack((y_train, y_pred_train)))
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model, _get_shap_explainer(model_name), xs_train, xs_test)
        np.savetxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Regression/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_svm(n_story: int, x: np.ndarray, y: np.ndarray, seed: int | None, shap_ana: bool = False, gs: bool=False):
    """训练SVM模型"""
    model_name = 'SVM'
    print(f'=========================== SVM ===========================')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.fit_transform(y_train)
    xs_test = x_scaler.transform(x_test)

    if n_story == 4:
        paras = {'C': 10, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    elif n_story == 8:
        paras = {'C': 10, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    elif n_story == 12:
        paras = {'C': 10, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    svr = SVR(
        coef0=0.0,
        shrinking=True,
        tol=0.001,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        **paras
    )  # 网格搜索结果: 
    model = MultiOutputRegressor(svr)
    
    # 带交叉验证的网格搜索
    if gs:
        C = [0.1, 1, 10]
        gamma = [0.001, 0.01, 0.1]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        degree = [2, 3, 4]
        param_grid = {
            'estimator__C': C,
            'estimator__gamma': gamma,
            'estimator__kernel': kernel,
            'estimator__degree': degree,
        }
        scorer = make_scorer(r2_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=VERBOSE, n_jobs=20)
        grid_search.fit(xs_train, ys_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(xs_train, ys_train)
    ys_pred_test = model.predict(xs_test).reshape(-1, len(LABEL_NAMES))
    ys_pred_train = model.predict(xs_train).reshape(-1, len(LABEL_NAMES))
    y_pred_test = y_scaler.inverse_transform(ys_pred_test)
    y_pred_train = y_scaler.inverse_transform(ys_pred_train)
    y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
    y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
    y_pred_test = np.where(y_pred_test > 1, 1, y_pred_test)
    y_pred_train = np.where(y_pred_train > 1, 1, y_pred_train)
    res = pd.DataFrame(columns=LABEL_NAMES+['global'],
                       index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
    for i in range(len(LABEL_NAMES)):
        label = LABEL_NAMES[i]  # 特征名
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        rmse_test = root_mean_squared_error(y_test[:, i], y_pred_test[:, i])
        rmse_train = root_mean_squared_error(y_train[:, i], y_pred_train[:, i])
        res.loc['R2_test', label] = r2_test
        res.loc['R2_train', label] = r2_train
        res.loc['RMSE_test', label] = rmse_test
        res.loc['RMSE_train', label] = rmse_train
    global_R2_test = r2_score(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_R2_train = r2_score(y_train.reshape(-1), y_pred_train.reshape(-1))
    global_RMSE_test = root_mean_squared_error(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_RMSE_train = root_mean_squared_error(y_train.reshape(-1), y_pred_train.reshape(-1))
    res.loc['R2_test', 'global'] = global_R2_test
    res.loc['R2_train', 'global'] = global_R2_train
    res.loc['RMSE_test', 'global'] = global_RMSE_test
    res.loc['RMSE_train', 'global'] = global_RMSE_train
    # _plot_scatter(ax, y_test, y_pred_test, model_name, res.loc['R2_test', label_to_plot])
    print(res)
    output = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }
    joblib.dump(output, f'models/regressors/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, res, x_test, x_train, np.column_stack((y_test, y_pred_test)), np.column_stack((y_train, y_pred_train)))
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model.predict, _get_shap_explainer(model_name), xs_train, xs_test)
        np.savetxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Regression/shap_features_{model_name}_{n_story}S.txt', shap_features)

def train_ann(n_story: int, x: np.ndarray, y: np.ndarray, seed: int | None, shap_ana: bool = False, gs: bool=False):
    """训练ANN模型"""
    model_name = 'ANN'
    print(f'=========================== ANN ===========================')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.fit_transform(y_train)
    xs_test = x_scaler.transform(x_test)

    if n_story == 4:
        paras = {'hidden_layer_sizes': (128, 64), 'learning_rate_init': 0.001}
    elif n_story == 8:
        paras = {'hidden_layer_sizes': (128, 64), 'learning_rate_init': 0.001}
    elif n_story == 12:
        paras = {'hidden_layer_sizes': (64, 32), 'learning_rate_init': 0.01}
    else:
        raise NotImplementedError(f'{n_story}-story is not supported.')
    model = MLPRegressor(
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
        **paras
    )  # 网格搜索结果: n_estimators=400, max_depth=20
    
    # 带交叉验证的网格搜索
    if gs:
        hidden_layer_sizes = [(8,), (16), (32), (16, 8), (32, 16), (64, 32), (128, 64)]
        learning_rate_init = [0.001, 0.01, 0.1, 1]
        param_grid = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'learning_rate_init': learning_rate_init,
        }
        scorer = make_scorer(r2_score)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=VERBOSE, n_jobs=16)
        grid_search.fit(xs_train, ys_train[:, 0])
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        model = grid_search.best_estimator_

    model.fit(xs_train, ys_train[:, 0])
    ys_pred_test = model.predict(xs_test).reshape(-1, len(LABEL_NAMES))
    ys_pred_train = model.predict(xs_train).reshape(-1, len(LABEL_NAMES))
    y_pred_test = y_scaler.inverse_transform(ys_pred_test)
    y_pred_train = y_scaler.inverse_transform(ys_pred_train)
    y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
    y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
    y_pred_test = np.where(y_pred_test > 1, 1, y_pred_test)
    y_pred_train = np.where(y_pred_train > 1, 1, y_pred_train)
    res = pd.DataFrame(columns=LABEL_NAMES+['global'],
                       index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
    for i in range(len(LABEL_NAMES)):
        label = LABEL_NAMES[i]  # 特征名
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        rmse_test = root_mean_squared_error(y_test[:, i], y_pred_test[:, i])
        rmse_train = root_mean_squared_error(y_train[:, i], y_pred_train[:, i])
        res.loc['R2_test', label] = r2_test
        res.loc['R2_train', label] = r2_train
        res.loc['RMSE_test', label] = rmse_test
        res.loc['RMSE_train', label] = rmse_train
    global_R2_test = r2_score(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_R2_train = r2_score(y_train.reshape(-1), y_pred_train.reshape(-1))
    global_RMSE_test = root_mean_squared_error(y_test.reshape(-1), y_pred_test.reshape(-1))
    global_RMSE_train = root_mean_squared_error(y_train.reshape(-1), y_pred_train.reshape(-1))
    res.loc['R2_test', 'global'] = global_R2_test
    res.loc['R2_train', 'global'] = global_R2_train
    res.loc['RMSE_test', 'global'] = global_RMSE_test
    res.loc['RMSE_train', 'global'] = global_RMSE_train
    # _plot_scatter(ax, y_test, y_pred_test, model_name, res.loc['R2_test', label_to_plot])
    print(res)
    output = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }
    joblib.dump(output, f'models/regressors/{model_name}_{n_story}S.joblib')
    _output(n_story, model_name, res, x_test, x_train, np.column_stack((y_test, y_pred_test)), np.column_stack((y_train, y_pred_train)))
    if shap_ana:
        shap_values, shap_features = _get_shap_values(model.predict, _get_shap_explainer(model_name), xs_train, xs_test)
        np.savetxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt', shap_values)
        np.savetxt(f'ml_output/Regression/shap_features_{model_name}_{n_story}S.txt', shap_features)

def plot_scatter(n_story: int, model_names: list[str], label_to_plot: str):
    """绘制散点图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, model_name in enumerate(model_names):
        x_test = pd.read_csv(f'ml_output/Regression/{model_name}_{n_story}S_x_test.csv', index_col=0)
        y_test = pd.read_csv(f'ml_output/Regression/{model_name}_{n_story}S_y_test.csv', index_col=0)
        data_actual = y_test[f'Actual_{label_to_plot}'].to_numpy()
        data_predicted = y_test[f'Predicted_{label_to_plot}'].to_numpy()
        R2 = r2_score(data_actual, data_predicted)
        ax: Axes = axes.flat[i]
        ax.plot([0, np.max(data_actual)*1.1], [0, np.max(data_actual)*1.1], color='red', label='1:1')
        ax.scatter(data_actual, data_predicted, label=label_to_plot)
        ax.set_title(f'{model_name} (R2={R2:.4f})')
        ax.set_xlabel('True')
        ax.set_ylabel('Predict')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'ml_output/Regression/Results_{n_story}S.png', dpi=600)
    plt.show()

def _output(
    n_story: int,
    model_name: str,
    res: pd.DataFrame,
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
    res.to_csv(f'ml_output/Regression/{model_name}_{n_story}S.csv')
    columns = ['Actual_' + name for name in LABEL_NAMES] + ['Predicted_' + name for name in LABEL_NAMES]
    df_x_test = pd.DataFrame(x_test, columns=feature_names)
    df_x_train = pd.DataFrame(x_train, columns=feature_names)
    df_y_test = pd.DataFrame(y_test, columns=columns)
    df_y_train = pd.DataFrame(y_train, columns=columns)
    df_x_test.to_csv(f'ml_output/Regression/{model_name}_{n_story}S_x_test.csv')
    df_x_train.to_csv(f'ml_output/Regression/{model_name}_{n_story}S_x_train.csv')
    df_y_test.to_csv(f'ml_output/Regression/{model_name}_{n_story}S_y_test.csv')
    df_y_train.to_csv(f'ml_output/Regression/{model_name}_{n_story}S_y_train.csv')

def plot_shap(n_story: int, feature_names: list[str], model_name: str, label_to_plot: str):
    """绘制shap图"""
    df_x = pd.read_csv(f'ml_output/Regression/{model_name}_{n_story}S_x_test.csv', index_col=0)
    df_y = pd.read_csv(f'ml_output/Regression/{model_name}_{n_story}S_y_test.csv', index_col=0)
    x_test = df_x.to_numpy()
    y_test = df_y[f'Actual_{label_to_plot}'].to_numpy()
    shap_values: np.ndarray = np.loadtxt(f'ml_output/Regression/shap_values_{model_name}_{n_story}S.txt')
    if len(LABEL_NAMES) > 1:
        idx = LABEL_NAMES.index(label_to_plot)
        shap_values = shap_values[:, :, idx]
    plt.title(f'{model_name} (Feature={label_to_plot})')
    shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
    plt.savefig(f'ml_output/Regression/shap_{model_name}_{n_story}S_{label_to_plot}.png', dpi=600)
    plt.show()
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance_df.to_csv(f'ml_output/Regression/{model_name}_{n_story}S_{label_to_plot}_importance.csv')
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.title(f'{model_name} (Feature={label_to_plot})')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'ml_output/Regression/{model_name}_{n_story}S_{label_to_plot}_importance.png', dpi=600)
    plt.show()

def train_all(n_story: int, x, y, seed: int | None, gs: bool):
    pd.set_option('display.float_format', '{:.4f}'.format)
    train_linear_regression(n_story, x, y, seed, shap_ana=False, gs=gs)
    train_knn(n_story, x, y, seed, shap_ana=False, gs=gs)
    train_random_forest(n_story, x, y, seed, shap_ana=False, gs=gs)
    train_xgboost(n_story, x, y, seed, shap_ana=True, gs=gs)
    train_svm(n_story, x, y, seed, shap_ana=False, gs=gs)
    train_ann(n_story, x, y, seed, shap_ana=False, gs=gs)

if __name__ == "__main__":
    """
    输入: R, lamda, a1, a2 (8- and 12-story), a3 (12-story)
    输出: loss1_mean
    """
    seed = 42
    gs = False  # 是否进行网格搜索
    for n_story in [4, 8, 12]:
        feature_names = FEATURE_NAMES[str(n_story)]
        data2 = pd.read_csv(f"data/data2_{n_story}S.csv", index_col=0)
        x = data2[feature_names].to_numpy().reshape(-1, len(feature_names))
        y = data2[LABEL_NAMES].to_numpy().reshape(-1, len(LABEL_NAMES))
        Path('ml_output/Regression').mkdir(parents=True, exist_ok=True)
        Path('models/regressors').mkdir(parents=True, exist_ok=True)
        train_all(n_story, x, y, seed, gs)
        plot_scatter(n_story, ['LR', 'KNN', 'RF', 'XGBoost', 'SVM', 'ANN'], 'loss1_mean')
        plot_shap(n_story, feature_names, 'XGBoost', label_to_plot='loss1_mean')
