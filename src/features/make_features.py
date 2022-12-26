import argparse
import warnings
from typing import Dict

import pandas as pd
from src.features.base import ConcatFeatures, RawFeatures
from src.features.table import ConcatGroupAggregation
from src.utils.io import load_config

warnings.filterwarnings("ignore")


def get_feature_transformer(config: Dict) -> ConcatFeatures:

    feature_transformers = [
        RawFeatures(**config["raw_features"]),
        ConcatGroupAggregation(**config["concat_group_aggregation"]),
    ]
    save_path = config["feature_path"]
    keep_columns = config["keep_columns"]
    return ConcatFeatures(feature_transformers, save_path, keep_columns)


def main(config_path):
    # Load Configuration.
    config = load_config(config_path)
    df = pd.read_pickle(config["processed_data_path"])
    feature_transformer = get_feature_transformer(config)
    _ = feature_transformer.fit_transform(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, default="config/features/baseline.yaml"
    )
    args = parser.parse_args()
    main(args.config_path)
