import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import numpy as np
import copy
from typing import Tuple, List


def load_dataset(dataset: str = 'animals') ->pd.DataFrame:
    if dataset == 'animals':
        return pd.read_excel('datasets/Animals.xls')
    elif dataset == 'stars':
        return sm.datasets.get_rdataset('starsCYG', 'robustbase', cache=True).data


def scatter_plot_animals(df: pd.DataFrame) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    names = df['Animal'].values
    body_w = df['Body Weight'].values
    brain_w = df['Brain Weight'].values

    for (bo_w, br_w) in zip(body_w, brain_w):
        ax.scatter(bo_w, br_w)
    ax.set_xlabel('Body weight (kg)')
    ax.set_ylabel('Brain weight (g)')
    ax.set_title('Body and brain weight dataset')

    for i, text in enumerate(names):
        ax.annotate(text, (body_w[i], brain_w[i]))
    plt.show()


def remove_outliers(chromosomes: np.ndarray, dataset: str, cook_distance: bool = False) -> List[np.ndarray]:
    fitness_function = []
    df = load_dataset(dataset)
    # chromosomes[0] = np.array([5, 6, 15, 24])
    dataset_dim = df.shape[0]
    for chromosome in chromosomes:

        if cook_distance:
            if dataset == 'animals':
                coef_w, intercept_w = fit_regression(df['Body Weight'].values, df['Brain Weight'].values)
            elif dataset == 'stars':
                coef_w, intercept_w = fit_regression(df['log.Te'].values, df['log.light'].values)
        for idx in chromosome:
            df.drop(idx, axis=0, inplace=True)

        if dataset == 'animals':
            coef, intercept = fit_regression(df['Body Weight'].values, df['Brain Weight'].values)
        elif dataset == 'stars':
            coef, intercept = fit_regression(df['log.Te'].values, df['log.light'].values)

        if cook_distance:
            if dataset == 'animals':
                fitness_function.append(calculate_cook_distance(df['Body Weight'].values, df['Brain Weight'].values,
                                                                coef_w, intercept_w, coef, intercept, dataset_dim))
            elif dataset == 'stars':
                fitness_function.append(calculate_cook_distance(df['log.Te'].values, df['log.light'].values,
                                                                coef_w, intercept_w, coef, intercept, dataset_dim))
        else:
            if dataset == 'animals':
                fitness_function.append(calculate_squares(df['Body Weight'].values, df['Brain Weight'].values,
                                                     coef, intercept))
            elif dataset == 'stars':
                fitness_function.append(calculate_squares(df['log.Te'].values, df['log.light'].values,
                                                          coef, intercept))
        df = load_dataset(dataset)
    return fitness_function


def calculate_squares(X: np.ndarray, y: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    y_preds = X * coef + intercept
    sum_squares = np.sum((y - y_preds)**2)
    return sum_squares


def calculate_cook_distance(X: np.ndarray, y: np.ndarray, coef_w: float,
                            intercept_w: float, coef: float, intercept: float, dim: int) -> np.ndarray:
    y_preds_w = X * coef_w + intercept_w
    y_preds = X * coef + intercept

    sum_difference = np.sum((y_preds_w - y_preds)**2)
    sum_squares = np.sum((y - y_preds)**2)
    sum_difference /= (dim * (sum_squares / dim - 2))
    return sum_difference


def scatter_plot_cyg(df: pd.DataFrame) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    temp = df['log.Te'].values
    light = df['log.light'].values

    for (t, l) in zip(temp, light):
        ax.scatter(t, l)
    ax.set_xlabel('Log of temperature')
    ax.set_ylabel('Log of light')
    ax.set_title('Cygnus dataset')
    plt.show()


def fit_regression(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, show: bool = False) -> Tuple[float, float]:
    reg = LinearRegression()
    reg_w = LinearRegression()

    reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    reg_w.fit(df['log.Te'].values.reshape(-1, 1), df['log.light'].values.reshape(-1, 1))

    y_preds = reg.predict(X.reshape(-1, 1))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for (xx, yy) in zip(X, y):
            ax.scatter(xx, yy)

        # ax.plot(X, (X * reg.coef_ + reg.intercept_).flatten(), 'k--',
        #         label=f'R2 score: {r2_score(y.reshape(-1, 1), y_preds):.2f}')

        ax.plot(X, (X * reg.coef_ + reg.intercept_).flatten(), 'k--',
                label=f'Sum of squares: {calculate_squares(X, y, reg.coef_, reg.intercept_):.2f}\nR2 score: '
                      f'{r2_score(y.reshape(-1, 1), y_preds):.2f}'
                      f'\nCook\'s distance: {calculate_cook_distance(X, y, reg_w.coef_, reg_w.intercept_, reg.coef_, reg.intercept_, dim=df.shape[0]):.2f}')
        # ax.set_xlabel('Body weight (kg)')
        # ax.set_ylabel('Brain weight (g)')
        ax.set_xlabel('Log of temperature')
        ax.set_ylabel('Log of light')
        ax.legend(loc='upper right')
        ax.set_title('Regression fit without outliers,\nCook\'s distance')
        plt.show()
    return reg.coef_, reg.intercept_


def calculate_ratio(df: pd.DataFrame) -> None:
    df['ratio'] = df['Brain Weight'] / df['Body Weight']
    print(df)


if __name__ == '__main__':
    df = load_dataset('stars')
    # scatter_plot_animals(df)
    # calculate_ratio(df)

    df_copy = df.copy(deep=True)

    # df = df[df['Animal'] != 'Brachiosaurus']
    # df = df[df['Animal'] != 'Dipliodocus']
    # df = df[df['Animal'] != 'Triceratops']
    # df = df[df['Animal'] != 'African elephant']
    # df = df[df['Animal'] != 'Asian elephant']
    # df = df[df['Animal'] != 'Giraffe']
    # df = df[df['Animal'] != 'Human']
    # df = df[df['Animal'] != 'Mountain beaver']
    # df = df[df['Animal'] != 'Guinea pig']

    df.drop(index=[0, 25, 22, 33], inplace=True)

    # coef, intercept = fit_regression(df_copy, df['Body Weight'].values, df['Brain Weight'].values, show=True)
    coef, intercept = fit_regression(df_copy, df['log.Te'].values, df['log.light'].values, show=True)
