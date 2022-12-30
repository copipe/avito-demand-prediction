import gc
from pathlib import Path
from typing import List, Union

import pandas as pd
from src.features.base import AbstractFeatureTransformer, ConcatSameClassFeatures


class GroupAggregation(AbstractFeatureTransformer):
    def __init__(
        self,
        save_path: Union[str, Path],
        group_keys: Union[str, List],
        target: str,
        agg: str,
        fillna: int,
        filter_train: bool,
        feature_name: str,
    ):
        super().__init__(save_path)
        if isinstance(group_keys, list):
            self.group_keys = group_keys
        else:
            self.group_keys = [group_keys]
        self.target = target
        self.agg = agg
        self.feature_name = feature_name
        self.fillna = fillna
        self.filter_train = filter_train
        self.name = f"{self.name}-{feature_name}"

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.filter_train:
            df_ = df[df["is_train"] == 1]
            feature = df_.groupby(self.group_keys)[self.target].agg(self.agg)
            del df_
            gc.collect()
        else:
            feature = df.groupby(self.group_keys)[self.target].agg(self.agg)
        feature = feature.reset_index()
        feature = feature.rename(columns={self.target: self.feature_name})
        feature = pd.merge(df[self.group_keys], feature, how="left", on=self.group_keys)
        feature = feature.fillna(self.fillna)
        feature = feature.drop(self.group_keys, axis=1)
        return feature


class ConcatGroupAggregation(ConcatSameClassFeatures):
    def __init__(
        self,
        configs: List,
        save_dir: Union[str, Path],
    ):
        super().__init__(configs, save_dir, GroupAggregation)


class DiffFeatures(AbstractFeatureTransformer):
    def __init__(
        self,
        save_path: str,
        left_feature: str,
        right_feature: str,
        feature_name: str,
    ):
        super().__init__(save_path)
        self.left = left_feature
        self.right = right_feature
        self.feature_name = feature_name

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feature = df[self.left] - df[self.right]
        feature = pd.DataFrame(feature, columns=[self.feature_name])
        return feature


class ConcatDiffFeatures(ConcatSameClassFeatures):
    def __init__(
        self,
        configs: List,
        save_dir: Union[str, Path],
    ):
        super().__init__(configs, save_dir, DiffFeatures)
