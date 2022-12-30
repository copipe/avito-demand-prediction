import gc
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.base import AbstractFeatureTransformer, ConcatSameClassFeatures


class TextBasicAggregation(AbstractFeatureTransformer):
    def __init__(
        self,
        save_path: str,
        text_column: str,
        feature_name: str,
    ):
        super().__init__(save_path)
        self.text_column = text_column
        self.feature_name = feature_name

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        n_chars = df[self.text_column].apply(len)
        n_words = df[self.text_column].apply(lambda t: len(t.split()))
        n_unique_words = df[self.text_column].apply(lambda t: len(set(t.split())))
        feature = pd.DataFrame(
            {
                f"n_chars_{self.feature_name}": n_chars,
                f"n_words_{self.feature_name}": n_words,
                f"n_unique_words_{self.feature_name}": n_unique_words,
            }
        )
        return feature


class ConcatTextBasicAggregation(ConcatSameClassFeatures):
    def __init__(
        self,
        configs: List,
        save_dir: Union[str, Path],
    ):
        super().__init__(configs, save_dir, TextBasicAggregation)


class TFIDFAggregation(AbstractFeatureTransformer):
    def __init__(
        self,
        save_path: str,
        tfidf_params: Dict,
        text_column: str,
        feature_name: str,
    ):
        super().__init__(save_path)
        self.tfidf_params = tfidf_params
        self.text_column = text_column
        self.feature_name = feature_name
        self.tfidf_params["stop_words"] = stopwords.words("russian")

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        tfidf = TfidfVectorizer(**self.tfidf_params)
        vectors = tfidf.fit_transform(df[self.text_column])
        max_vectors = np.array(vectors.max(axis=1).todense()).ravel()
        sum_vectors = np.array(vectors.sum(axis=1)).ravel()
        n_nonzeros = np.array((vectors != 0).sum(axis=1)).ravel()
        avg_vectors = sum_vectors / (n_nonzeros + 1e-6)
        feature = pd.DataFrame(
            {
                f"max_{self.feature_name}": max_vectors,
                f"sum_{self.feature_name}": sum_vectors,
                f"n_nonzeros_{self.feature_name}": n_nonzeros,
                f"avg_{self.feature_name}": avg_vectors,
            }
        )

        return feature


class ConcatTFIDFAggregation(ConcatSameClassFeatures):
    def __init__(
        self,
        configs: List,
        save_dir: Union[str, Path],
    ):
        super().__init__(configs, save_dir, TFIDFAggregation)


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
        self.tfidf_params["stop_words"] = stopwords.words("russian")

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        tfidf = TfidfVectorizer(**self.tfidf_params)
        tsvd = TruncatedSVD(**self.svd_params)
        feature = tfidf.fit_transform(df[self.text_column])
        feature = tsvd.fit_transform(feature)
        columns = [f"{self.feature_name}_d{i}" for i in range(feature.shape[1])]
        feature = pd.DataFrame(feature, columns=columns)
        del tfidf, tsvd
        gc.collect()
        return feature


class ConcatTFIDFSVD(ConcatSameClassFeatures):
    def __init__(
        self,
        configs: List,
        save_dir: Union[str, Path],
    ):
        super().__init__(configs, save_dir, TFIDFSVD)
