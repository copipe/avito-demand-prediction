import argparse
import gc
import pickle
from pathlib import Path
from typing import Dict, Union

import cudf
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(input_dir: Path) -> pd.DataFrame:
    train = cudf.read_csv(input_dir / "train.csv").to_pandas()
    test = cudf.read_csv(input_dir / "test.csv").to_pandas()

    train["is_train"] = 1
    test["is_train"] = 0
    df = pd.concat([train, test]).reset_index(drop=True)

    del train, test
    gc.collect()
    return df


def id2int(df: pd.DataFrame, output_dir: Union[Path, None] = None) -> pd.DataFrame:
    uid2int = {uid: i for i, uid in enumerate(df["user_id"].unique())}
    iid2int = {iid: i for i, iid in enumerate(df["item_id"].unique())}
    df["user_id"] = df["user_id"].map(uid2int)
    df["item_id"] = df["item_id"].map(iid2int)
    if output_dir:
        with open(output_dir / "uid2int.pickle", "wb") as f:
            pickle.dump(uid2int, f)

        with open(output_dir / "iid2int.pickle", "wb") as f:
            pickle.dump(iid2int, f)
    return df


def groupkfold(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    df["kfold"] = -1
    kfold = GroupKFold(n_splits=n_splits).split(df, groups=df["user_id"])
    all_index = df.index
    is_train = df["is_train"] == 1
    for k_index, (_, fold_index) in enumerate(kfold):
        fold_index = np.isin(all_index, fold_index)
        df.loc[is_train & fold_index, "kfold"] = k_index
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        "title",
        "description",
    ]
    for text_col in text_cols:
        df[text_col] = df[text_col].str.lower().fillna("")
    return df


def clean_number(df: pd.DataFrame) -> pd.DataFrame:
    df["image_score"] = df["image_top_1"].fillna(-1).astype(int)
    df["price"] = np.log1p(df["price"]).fillna(0)
    df["deal_probability"] = df["deal_probability"].fillna(-1)
    df = df.drop("image_top_1", axis=1)
    return df


def encode_category(df: pd.DataFrame) -> pd.DataFrame:
    df["region_id"] = LabelEncoder().fit_transform(df["region"])
    df["category_id_1"] = LabelEncoder().fit_transform(df["parent_category_name"])
    df["category_id_2"] = LabelEncoder().fit_transform(df["category_name"])
    df["user_type"] = LabelEncoder().fit_transform(df["user_type"])
    return df


def concat_text(df: pd.DataFrame) -> pd.DataFrame:
    df["region"] = df["region"] + " " + df["city"]
    df["category_name"] = df["parent_category_name"] + " " + df["category_name"]
    df["param"] = df["param_1"] + " " + df["param_2"] + " " + df["param_3"]

    drop_cols = ["city", "parent_category_name", "param_1", "param_2", "param_3"]
    df = df.drop(drop_cols, axis=1)
    return df


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "user_id",
        "item_id",
        "deal_probability",
        "is_train",
        "kfold",
        "price",
        "item_seq_number",
        "image_score",
        "user_type",
        "region_id",
        "category_id_1",
        "category_id_2",
        "title",
        "description",
        "param",
        "region",
        "category_name",
        "activation_date",
        "image",
    ]
    df = df.reset_index(drop=True)
    df = df[cols].copy()
    return df


def main(config_path: str) -> None:
    # Load config file.
    config = load_config(config_path)
    input_dir = Path(config["input_dir"])
    output_path = Path(config["output_path"])
    n_splits = config["n_splits"]

    # Preprocess data.
    df = load_data(input_dir)
    df = id2int(df, output_path.parent)
    df = groupkfold(df, n_splits)
    df = clean_text(df)
    df = clean_number(df)
    df = encode_category(df)
    df = concat_text(df)
    df = sort_columns(df)

    # Save processed data.
    df.to_pickle(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, default="config/preprocess/baseline.yaml"
    )
    args = parser.parse_args()
    main(args.config_path)
