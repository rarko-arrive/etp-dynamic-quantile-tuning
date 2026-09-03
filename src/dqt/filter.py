import polars as pl

from dqt import resolve_data_dir
from dqt.score.constants import ID_COL

# df = pl.scan_parquet(resolve_data_dir() / "etp/dqt_vs_non_adjusted.parquet")

df = pl.scan_parquet(resolve_data_dir() / "features.parquet")
# df.head().collect()
# df.filter(pl.col("load_type").is_in(("DRY", "REEFER")))

# df.filter(pl.col("load_type").is_in(("DRY", "REEFER"))).collect()

ix_rm_equip = ~pl.col("load_type").is_in(("DRY", "REEFER"))
ix_bad_bounce = pl.col("is_bad_bounce").is_not_null()


df_rm = df.filter(ix_rm_equip | ix_bad_bounce).select(ID_COL)
df_clean = df.join(df_rm, on="loadnumber", how="anti")

df_clean.collect()