import pandas as pd
import pathway as pw
from pathway.stdlib.temporal import Direction

def asof_align_macro(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    suffix: str = "macro_",
    direction: Direction = Direction.BACKWARD,
) -> pd.DataFrame:
    """
    Align macro time series onto per-ticker rows using Pathway asof_join.

    Args:
        df: DataFrame with at least ['kdcode', 'dt'] (dt can be str or datetime-like).
        macro_df: DataFrame with a 'dt' column and macro feature columns.
        suffix: Prefix applied to macro feature columns to avoid collisions.
        direction: Pathway asof direction (BACKWARD/NEAREST/FORWARD).

    Returns:
        DataFrame with macro columns appended. Falls back to pandas.merge_asof on failure.
    """
    if macro_df.empty or df.empty:
        return df

    original_dt_dtype = df["dt"].dtype
    left = df.copy()
    left["dt"] = pd.to_datetime(left["dt"])

    right = macro_df.copy()
    if "datetime" in right.columns and "dt" not in right.columns:
        right = right.rename(columns={"datetime": "dt"})
    right["dt"] = pd.to_datetime(right["dt"])
    macro_cols = [c for c in right.columns if c != "dt"]
    right = right.rename(columns={c: f"{suffix}{c}" for c in macro_cols})

    try:
        left_tbl = pw.debug.table_from_pandas(left)
        right_tbl = pw.debug.table_from_pandas(right)

        select_kwargs = {col: getattr(left_tbl, col) for col in left.columns}
        for col in right.columns:
            if col == "dt":
                continue
            select_kwargs[col] = getattr(right_tbl, col)

        joined = (
            left_tbl.asof_join(
                right_tbl,
                left_tbl.dt,
                right_tbl.dt,
                how=pw.JoinMode.LEFT,
                direction=direction,
            ).select(**select_kwargs)
        )
        pdf = pw.debug.table_to_pandas(joined)
    except Exception as exc:
        print(f"[warn] Pathway asof_join failed ({exc}); falling back to pandas.merge_asof.")
        pdf = pd.merge_asof(
            left.sort_values("dt"),
            right.sort_values("dt"),
            on="dt",
            direction="backward",
        )

    duplicate_dt_cols = [c for c in pdf.columns if c.startswith("dt_")]
    pdf = pdf.drop(columns=duplicate_dt_cols)

    if pd.api.types.is_object_dtype(original_dt_dtype):
        pdf["dt"] = pdf["dt"].dt.strftime("%Y-%m-%d")

    return pdf