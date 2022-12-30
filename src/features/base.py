import gc
from pathlib import Path
from typing import List, Union

import pandas as pd


class AbstractFeatureTransformer:
    def __init__(self, save_path: Union[str, Path]):
        self.name = self.__class__.__name__
        self.save_path = Path(save_path)

    def fit(self, df: pd.DataFrame) -> None:
        if not self.save_path.exists():
            self._fit(df)

    def _fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.save_path.exists():
            return self._load_feature()
        else:
            feature = self._transform(df)
            self._save_feature(feature)
            return feature

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def _save_feature(self, df: pd.DataFrame) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(self.save_path)
        print(f"Save {self.name} feature to '{str(self.save_path)}'.")

    def _load_feature(self) -> pd.DataFrame:
        return pd.read_pickle(self.save_path)


class RawFeatures(AbstractFeatureTransformer):
    def __init__(
        self,
        columns: List[str],
        save_path: Union[str, Path],
    ):
        super().__init__(save_path)
        self.columns = columns

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.columns]
        return features


class ConcatFeatures(AbstractFeatureTransformer):
    def __init__(
        self,
        feature_transformers: List[AbstractFeatureTransformer],
        save_path: Union[str, Path],
        keep_columns: List[str] = [],
    ):
        super().__init__(save_path)
        self.feature_transformers = feature_transformers
        self.keep_columns = keep_columns

    def _fit(self, df: pd.DataFrame):
        for feature_transformer in self.feature_transformers:
            feature_transformer.fit(df)
            gc.collect()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = []
        for feature_transformer in self.feature_transformers:
            feature = feature_transformer.transform(df)
            for col in feature.columns:
                df[col] = feature[col]
                feature_columns.append(col)
            del feature_transformer, feature
            gc.collect()

        features = df[self.keep_columns + feature_columns]
        return features


class ConcatSameClassFeatures(ConcatFeatures):
    def __init__(
        self,
        configs: List,
        save_dir: Union[str, Path],
        feature_transformer: AbstractFeatureTransformer,
    ):
        self.configs = configs
        self.save_dir = Path(save_dir)
        self.feature_transformer = feature_transformer
        feature_transformers = self._get_feature_transformers()
        super().__init__(feature_transformers, self.save_dir / "concat.pickle")

    def _get_feature_transformers(self):
        transformers = []
        for config in self.configs:
            config["save_path"] = self.save_dir / f"{config['feature_name']}.pickle"
            transformers.append(self.feature_transformer(**config))
        return transformers
