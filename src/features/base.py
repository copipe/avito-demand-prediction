import gc
from pathlib import Path
from typing import List

import pandas as pd


class AbstractFeatureTransformer:
    def __init__(self, save_path: str):
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
        save_path: str,
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
        save_path: str,
        keep_columns: List[str] = [],
    ):
        super().__init__(save_path)
        self.feature_transformers = feature_transformers
        self.keep_columns = keep_columns
        self.dummy_columns = ["item_id"]

    def _fit(self, df: pd.DataFrame):
        for feature_transformer in self.feature_transformers:
            print(f"{feature_transformer.name} fiting...")
            feature_transformer.fit(df)
            gc.collect()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.keep_columns:
            features = df[self.keep_columns]
        else:
            features = df[self.dummy_columns]

        for feature_transformer in self.feature_transformers:
            print(f"{feature_transformer.name} transforming...")
            feature = feature_transformer.transform(df)
            for col in feature.columns:
                features[col] = feature[col]
            del feature_transformer, feature
            gc.collect()

        if not self.keep_columns:
            features = features.drop(self.dummy_columns, axis=1)
        return features
