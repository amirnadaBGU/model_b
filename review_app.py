#!/usr/bin/env python3
"""
review_app.py – Streamlit crop review and labelling tool.
Run with:  streamlit run review_app.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
CROPS_BASE    = Path("crops")
CLASS_OPTIONS = ["fish", "partial_fish", "background"]
DATASETS      = ["data12", "data15", "data25"]
SPLITS        = ["valid", "train", "test"]
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Crop Review", layout="wide")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"label": str, "review_status": str})
    if "review_status" not in df.columns:
        df["review_status"] = "pending"
    df["label"]         = df["label"].fillna("").astype(str)
    df["review_status"] = df["review_status"].fillna("pending").astype(str)
    return df


def save_df(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False)


def effective_class(row: pd.Series) -> str:
    """Return user label if set, otherwise fall back to model class_name."""
    return row["label"] if row["label"] in CLASS_OPTIONS else row["class_name"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown("""
    <style>
    [data-testid="stImage"] img {
        max-height: 68vh;
        width: auto !important;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Crop Review")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Dataset")
        dataset = st.selectbox("Dataset", DATASETS, key="sb_dataset")
        split   = st.selectbox("Split",   SPLITS,   key="sb_split")

        csv_path = CROPS_BASE / dataset / split / "labels.csv"
        img_dir  = CROPS_BASE / dataset / split

        if not csv_path.exists():
            st.error(f"labels.csv not found:\n{csv_path}")
            st.stop()

        df = load_df(csv_path)

        st.divider()
        st.header("Filters")
        status_filter = st.multiselect(
            "Review status",
            options=["pending", "approved", "rejected"],
            default=["pending", "approved", "rejected"],
            key="flt_status",
        )
        class_filter = st.multiselect(
            "Effective class",
            options=CLASS_OPTIONS,
            default=CLASS_OPTIONS,
            key="flt_class",
        )

        st.divider()
        st.header("View")
        view_mode = st.radio("Mode", ["Single", "Grid"], key="view_mode")
        n_cols = (
            st.slider("Columns", 2, 6, 4, key="grid_cols")
            if view_mode == "Grid"
            else 4
        )

        st.divider()
        total    = len(df)
        reviewed = int((df["review_status"] != "pending").sum())
        st.metric("Reviewed", f"{reviewed} / {total}")
        st.progress(reviewed / total if total > 0 else 0.0)

    # ── Apply filters ──────────────────────────────────────────────────────────
    eff  = df.apply(effective_class, axis=1)
    mask = df["review_status"].isin(status_filter) & eff.isin(class_filter)
    # reset_index() moves the original integer index into the "index" column
    fdf  = df[mask].copy().reset_index()

    if fdf.empty:
        st.info("No crops match the current filters.")
        return

    if view_mode == "Single":
        _single_view(fdf, df, csv_path, img_dir)
    else:
        _grid_view(fdf, df, csv_path, img_dir, n_cols)


# ── Single view ────────────────────────────────────────────────────────────────

def _single_view(
    fdf: pd.DataFrame, df: pd.DataFrame, csv_path: Path, img_dir: Path
) -> None:
    if "nav_idx" not in st.session_state:
        st.session_state.nav_idx = 0
    # Clamp in case filter change shrinks the list
    st.session_state.nav_idx = max(0, min(st.session_state.nav_idx, len(fdf) - 1))
    i = st.session_state.nav_idx

    c_prev, c_pos, c_next = st.columns([1, 6, 1])
    with c_prev:
        if st.button("← Prev", disabled=(i == 0)):
            st.session_state.nav_idx -= 1
            st.rerun()
    with c_pos:
        st.markdown(
            f"<p style='text-align:center;margin-top:6px'><b>{i+1} / {len(fdf)}</b></p>",
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button("Next →", disabled=(i == len(fdf) - 1)):
            st.session_state.nav_idx += 1
            st.rerun()

    _crop_card(fdf.iloc[i], df, csv_path, img_dir, key_prefix=f"s{i}")


def _crop_card(
    row: pd.Series,
    df: pd.DataFrame,
    csv_path: Path,
    img_dir: Path,
    key_prefix: str,
) -> None:
    orig_idx  = row["index"]          # integer index in the full df
    crop_path = img_dir / row["crop_filename"]

    col_img, col_meta = st.columns([2, 1])

    with col_img:
        if crop_path.exists():
            st.image(str(crop_path), use_column_width=True)
        else:
            st.warning(f"File not found:\n{crop_path}")

    with col_meta:
        icon = {"approved": "✅", "rejected": "❌"}.get(row["review_status"], "⏳")
        st.markdown(f"**Status:** {icon} {row['review_status']}")
        st.markdown(f"**File:** `{row['crop_filename']}`")
        st.markdown(f"**Source:** `{row['image_name']}`")
        st.markdown(f"**Model class:** {row['class_name']} (id {int(row['class_id'])})")
        st.markdown(f"**Confidence:** {row['confidence']:.4f}")

        cur = row["label"] if row["label"] in CLASS_OPTIONS else row["class_name"]
        new_label = st.selectbox(
            "Assign label",
            CLASS_OPTIONS,
            index=CLASS_OPTIONS.index(cur) if cur in CLASS_OPTIONS else 0,
            key=f"{key_prefix}_lbl",
        )

        c_a, c_r = st.columns(2)
        with c_a:
            if st.button("✅ Approve", key=f"{key_prefix}_app"):
                df.at[orig_idx, "label"]         = new_label
                df.at[orig_idx, "review_status"] = "approved"
                save_df(df, csv_path)
                st.rerun()
        with c_r:
            if st.button("❌ Reject", key=f"{key_prefix}_rej"):
                df.at[orig_idx, "label"]         = "background"
                df.at[orig_idx, "review_status"] = "rejected"
                save_df(df, csv_path)
                st.rerun()


# ── Grid view ──────────────────────────────────────────────────────────────────

def _grid_view(
    fdf: pd.DataFrame,
    df: pd.DataFrame,
    csv_path: Path,
    img_dir: Path,
    n_cols: int,
) -> None:
    for row_start in range(0, len(fdf), n_cols):
        chunk       = fdf.iloc[row_start : row_start + n_cols]
        col_widgets = st.columns(n_cols)

        for col_widget, (_, row) in zip(col_widgets, chunk.iterrows()):
            orig_idx  = row["index"]
            crop_path = img_dir / row["crop_filename"]

            with col_widget:
                if crop_path.exists():
                    st.image(str(crop_path), use_column_width=True)
                else:
                    st.caption("(file missing)")

                icon = {"approved": "✅", "rejected": "❌"}.get(row["review_status"], "⏳")
                st.caption(f"{icon} {row['class_name']} | conf {row['confidence']:.2f}")

                cur = row["label"] if row["label"] in CLASS_OPTIONS else row["class_name"]
                new_label = st.selectbox(
                    "Label",
                    CLASS_OPTIONS,
                    index=CLASS_OPTIONS.index(cur) if cur in CLASS_OPTIONS else 0,
                    key=f"g{orig_idx}_lbl",
                    label_visibility="collapsed",
                )

                c_a, c_r = st.columns(2)
                with c_a:
                    if st.button("✅", key=f"g{orig_idx}_app"):
                        df.at[orig_idx, "label"]         = new_label
                        df.at[orig_idx, "review_status"] = "approved"
                        save_df(df, csv_path)
                        st.rerun()
                with c_r:
                    if st.button("❌", key=f"g{orig_idx}_rej"):
                        df.at[orig_idx, "label"]         = "background"
                        df.at[orig_idx, "review_status"] = "rejected"
                        save_df(df, csv_path)
                        st.rerun()


if __name__ == "__main__":
    main()
