from typing import Dict

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.base import AbstractFeatureTransformer


class TFIDFSVD(AbstractFeatureTransformer):
    def __init__(
        self,
        save_path: str,
        tfidf_params: Dict,
        svd_params: Dict,
        text_column: str,
        feature_name: str,
    ):
        super().__init__(save_path)
        self.tfidf_params = tfidf_params
        self.svd_params = svd_params
        self.text_column = text_column
        self.feature_name = feature_name

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        tfidf = TfidfVectorizer(**self.tfidf_params)
        tsvd = TruncatedSVD(**self.svd_params)
        feature = tfidf.fit_transform(df[self.text_column])
        feature = tsvd.fit_transform(feature)
        columns = [f"{self.feature_name}_d{i}" for i in range(feature.shape[1])]
        feature = pd.DataFrame(feature, columns=columns)
        return feature
