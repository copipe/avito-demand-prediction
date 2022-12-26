import gc
from typing import List, Union

import pandas as pd
from src.features.base import AbstractFeatureTransformer, ConcatFeatures


class GroupAggregation(AbstractFeatureTransformer):
    def __init__(
        self,
        save_path: str,
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


class ConcatGroupAggregation(ConcatFeatures):
    def __init__(
        self,
        group_aggregation_configs: List,
        save_path: str,
    ):
        self.group_aggregation_configs = group_aggregation_configs
        feature_transformers = self._get_feature_transformers()
        super().__init__(feature_transformers, save_path)

    def _get_feature_transformers(self):
        transformers = []
        for config in self.group_aggregation_configs:
            transformers.append(GroupAggregation(**config))
        return transformers
