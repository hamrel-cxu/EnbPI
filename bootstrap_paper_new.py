import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from numpy.random import default_rng
from numba import jit


class DataConfig(BaseModel):
    data_name: str
    filename: str
    max_data_size: int = 10000
    target_column: str


class DataPreparation:
    def __init__(self, config: DataConfig):
        self.config = config
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        data_df = pd.read_csv(self.config.filename, skiprows=2)
        if self.config.data_name == 'solar':
            data_df.drop(columns=data_df.columns[0:5], inplace=True)
            data_df.drop(columns="Unnamed: 13", inplace=True)
            data_df = data_df.iloc[: min(
                self.config.max_data_size, data_df.shape[0]), :]
        return data_df

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        data_x = self.df.drop(self.config.target_column, axis=1)
        data_y = self.df[self.config.target_column]
        data_y = data_y.shift(-1)
        data_y.dropna(inplace=True)
        data_x.drop(data_x.tail(1).index, inplace=True)
        return data_x.to_numpy(), data_y.to_numpy()

    def split_data(self, data_x: np.ndarray, data_y: np.ndarray, train_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = data_x[:train_size, :]
        X_test = data_x[train_size:, :]
        y_train = data_y[:train_size]
        y_test = data_y[train_size:]
        return X_train, y_train, X_test, y_test


class ARTransformer:
    @staticmethod
    @jit(nopython=True)
    def one_dimen_transform(y_train: np.ndarray, y_test: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(y_train)
        n1 = len(y_test)
        X_train = np.zeros((n - d, d))  # from d+1,...,n
        X_test = np.zeros((n1, d))  # from n-d,...,n+n1-d
        for i in range(n - d):
            X_train[i, :] = y_train[i: i + d]
        for i in range(n1):
            if i < d:
                X_test[i, :] = np.r_[y_train[n - d + i:], y_test[:i]]
            else:
                X_test[i, :] = y_test[i - d: i]
        y_train = y_train[d:]
        return X_train, X_test, y_train, y_test


class BootstrapSyntheticData:
    def __init__(self, block_length: int, B: int):
        self.block_length = block_length
        self.B = B
        self.rng = default_rng()

    def generate_bootstrap_samples(self, n: int, m: int) -> np.ndarray:
        samples_idx = np.zeros((self.B, m), dtype=int)
        for b in range(self.B):
            sample_idx = np.random.choice(a=n, size=m, replace=True)
            samples_idx[b, :] = sample_idx
        return samples_idx


class DataBootstrap(BootstrapSyntheticData):
    def _id_bootstrap(self, n: int, rng_integers, n_blocks: int, nexts: np.ndarray, last_block: int) -> np.ndarray:
        blocks = rng_integers(low=0, high=last_block,
                              size=(n_blocks, 1), dtype=int)
        _id = (blocks + nexts).ravel()
        return _id

    def prepare_bootstrap(self, y_train: np.ndarray, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.block_length
        n = len(y_train)
        blocks_starts = np.arange(0, n - d, d)
        nexts = np.arange(d)
        last_block = len(blocks_starts)
        rng_integers = self.rng.integers
        n_blocks = int(np.ceil(n / d))
        _id = self._id_bootstrap(n, rng_integers, n_blocks, nexts, last_block)
        return X_train[_id], y_train[_id]


class Model:
    def __init__(self, model_name: str, bootstrap: BootstrapSyntheticData):
        self.model_name = model_name
        self.bootstrap = bootstrap

    def fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> list:
        if self.model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
        elif self.model_name == 'ExtraTreesRegressor':
            model = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
        elif self.model_name == 'RidgeCV':
            model = RidgeCV()
        elif self.model_name == 'LassoCV':
            model = LassoCV()

        bootstrap_samples = self.bootstrap.generate_bootstrap_samples(
            len(y_train), len(y_train))
        models = []
        for sample in bootstrap_samples:
            sample_X_train = X_train[sample]
            sample_y_train = y_train[sample]
            model.fit(sample_X_train, sample_y_train)
            models.append(model)
        return models

    def predict(self, models: list, X_test: np.ndarray) -> np.ndarray:
        predictions = [model.predict(X_test) for model in models]
        return np.median(predictions, axis=0)


# Example usage
data_config = DataConfig(
    data_name='solar',
    filename='solar.csv',
    max_data_size=10000,
    target_column='column_name'
)
data_preparation = DataPreparation(data_config)
X, y = data_preparation.prepare_data()
X_train, y_train, X_test, y_test = data_preparation.split_data(
    X, y, train_size=8000)

ar_transformer = ARTransformer()
X_train, X_test, y_train, y_test = ar_transformer.one_dimen_transform(
    y_train, y_test, d=2)

bootstrap = DataBootstrap(block_length=100, B=100)
bootstrap_X_train, bootstrap_y_train = bootstrap.prepare_bootstrap(
    y_train, X_train)

model = Model(model_name='RandomForestRegressor', bootstrap=bootstrap)
models = model.fit_model(bootstrap_X_train, bootstrap_y_train)
predictions = model.predict(models, X_test)

# evaluate the predictions
