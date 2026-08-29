import polars as pl
from sklearn.model_selection import train_test_split
from src import config as C


def build_splits():
    df = pl.read_parquet(C.SAMPLE_PARQUET).to_pandas()

    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["product"], random_state=C.SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["product"], random_state=C.SEED
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")

    out = pl.from_pandas(
        __import__("pandas").concat([train, val, test], ignore_index=True)
    )
    out.write_parquet(C.SPLITS_PARQUET)

    print(out.group_by("split").len().sort("split"))
    print(f"\nwrote {C.SPLITS_PARQUET}")
    print("\nescalation rate by split:")
    print(out.group_by("split").agg(pl.col("escalated").mean()).sort("split"))


def load_split(name):
    df = pl.read_parquet(C.SPLITS_PARQUET)
    return df.filter(pl.col("split") == name).to_pandas()


if __name__ == "__main__":
    build_splits()