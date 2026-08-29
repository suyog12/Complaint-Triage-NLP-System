import polars as pl
from src import config as C


def check():
    df = pl.read_parquet(C.SPLITS_PARQUET)
    print("rows:", len(df), "unique narratives:", df["narrative"].n_unique())

    dupes = (df.group_by("narrative")
               .agg(pl.len().alias("n"),
                    pl.col("product").n_unique().alias("products"),
                    pl.col("split").n_unique().alias("splits"))
               .filter(pl.col("n") > 1)
               .sort("n", descending=True))

    print("duplicated narratives:", len(dupes))
    print("crossing splits:", dupes.filter(pl.col("splits") > 1).height)
    print(dupes.head(10))


if __name__ == "__main__":
    check()