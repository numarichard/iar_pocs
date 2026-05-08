"""Streamlit app for the table-view POC.

Sidebar nav:
  - Platform: render a spec (consumer side of the loop)
  - Feature:  author a spec interactively (producer side of the loop)

Schema versions
---------------
v1 (legacy): spec lists output columns, source is already at display grain.
v2:          spec lists output columns + per-column `aggregation`, declares a
             `period` window, and source is raw event-grain data. The renderer
             windows by `period.field`, groups by `grain`, applies each
             column's aggregation, and formats the result by column `type`.

The Feature (authoring) view supports v1 only; loading a v2 spec puts that
view in read-only mode so aggregation metadata survives the round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

POC_ROOT = Path(__file__).parent
CONFIG_PATH = POC_ROOT / "config" / "advisor_attention.yaml"

GRAIN_OPTIONS = [
    "service_advisor",
    "bdc_agent",
    "customer",
    "conversation",
    "rooftop",
    "(custom)",
]

COLUMN_TYPES = ["string", "integer", "float", "datetime", "boolean"]
VIZ_TYPES = ["table"]
FEATURE_AUTHORING_SCHEMA_VERSION = 1


# ---------- Loaders ----------

def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def save_config_if_changed(spec: dict) -> bool:
    """Write spec to disk only if it differs semantically from what's there.
    Keeps comments intact until the first real edit.
    """
    on_disk = load_config() if CONFIG_PATH.exists() else None
    if on_disk == spec:
        return False
    CONFIG_PATH.write_text(yaml.safe_dump(spec, sort_keys=False))
    return True


def load_dataframe(source: dict) -> pd.DataFrame:
    return pd.read_csv(POC_ROOT / source["path"])


def apply_sort(df: pd.DataFrame, sort_spec: list[dict] | None) -> pd.DataFrame:
    if not sort_spec:
        return df
    by = [s["column"] for s in sort_spec]
    ascending = [s.get("direction", "asc") == "asc" for s in sort_spec]
    return df.sort_values(by=by, ascending=ascending)


# ---------- v2 aggregation engine ----------

def _coerce_for_compare(series: pd.Series, value: Any) -> tuple[pd.Series, Any]:
    """Make CSV-loaded series comparable to YAML-loaded scalar.

    YAML booleans come in as Python bools; the CSV column is "true"/"false"
    strings unless we coerce. Same defensiveness for numbers.
    """
    if isinstance(value, bool):
        if series.dtype == object:
            return series.astype(str).str.lower(), "true" if value else "false"
        return series, value
    return series, value


def apply_predicate(df: pd.DataFrame, predicate: dict) -> pd.DataFrame:
    """Filter `df` by a recursive predicate.

    Supported shapes:
      {column: <c>, equals: <v>}
      {all: [<predicate>, ...]}   (logical AND)
    """
    if "all" in predicate:
        result = df
        for sub in predicate["all"]:
            result = apply_predicate(result, sub)
        return result
    if "column" in predicate and "equals" in predicate:
        col = predicate["column"]
        if col not in df.columns:
            return df.iloc[0:0]
        series, val = _coerce_for_compare(df[col], predicate["equals"])
        return df[series == val]
    raise ValueError(f"unsupported predicate: {predicate}")


def _pct_where(df: pd.DataFrame, predicate: dict) -> float | None:
    if not len(df):
        return None
    return len(apply_predicate(df, predicate)) / len(df) * 100


def _trend_label(current: float | None, prior: float | None, threshold_pp: float) -> str:
    if current is None or prior is None:
        return "—"
    delta = current - prior
    if abs(delta) < threshold_pp:
        return "Stable"
    return "↑ Worse" if delta > 0 else "↓ Better"


def _aggregate_group(
    rows: pd.DataFrame,
    prior_rows: pd.DataFrame,
    columns: list[dict],
    trend_threshold_pp: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        cid = col["id"]
        agg = col.get("aggregation") or {}
        t = agg.get("type")

        if t == "group_key":
            continue
        elif t == "count":
            out[cid] = len(rows)
        elif t == "count_where":
            out[cid] = len(apply_predicate(rows, agg["where"]))
        elif t == "pct_where":
            out[cid] = _pct_where(rows, agg["where"]) or 0
        elif t == "trend_pct_where":
            cur = _pct_where(rows, agg["where"])
            pri = _pct_where(prior_rows, agg["where"]) if prior_rows is not None else None
            out[cid] = _trend_label(cur, pri, trend_threshold_pp)
        elif t == "avg":
            vals = pd.to_numeric(rows[agg["column"]], errors="coerce").dropna()
            mean = float(vals.mean()) if len(vals) else 0.0
            if agg.get("output_unit") == "minutes":
                mean = mean / 60
            out[cid] = mean
        elif t == "mode":
            scope = apply_predicate(rows, agg["where"]) if "where" in agg else rows
            target = scope[agg["column"]] if agg["column"] in scope.columns else pd.Series(dtype=object)
            modes = target.mode()
            out[cid] = modes.iloc[0] if len(modes) else ""
        else:
            out[cid] = None
    return out


def _format_value(value: Any, col_type: str) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if col_type == "percent":
        return f"{int(round(float(value)))}%"
    if col_type == "duration_minutes":
        return f"{int(round(float(value)))}m"
    if col_type == "integer":
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return value
    return value


def aggregate_v2(spec: dict, df: pd.DataFrame, period_days: int) -> pd.DataFrame:
    """Run the v2 aggregation pipeline. Returns a DataFrame at `grain`."""
    period_cfg = spec["period"]
    field = period_cfg["field"]
    threshold = float(period_cfg.get("trend_threshold_pp", 5))

    df = df.copy()
    df[field] = pd.to_datetime(df[field])
    latest = df[field].max()
    cutoff = latest - pd.Timedelta(days=period_days)
    prior_cutoff = cutoff - pd.Timedelta(days=period_days)

    current = df[df[field] > cutoff]
    prior = df[(df[field] > prior_cutoff) & (df[field] <= cutoff)]

    grain_col = next(
        c["id"]
        for c in spec["columns"]
        if (c.get("aggregation") or {}).get("type") == "group_key"
    )

    rows: list[dict] = []
    for name, group in current.groupby(grain_col, sort=False):
        prior_group = prior[prior[grain_col] == name]
        agg_row = _aggregate_group(group, prior_group, spec["columns"], threshold)
        agg_row[grain_col] = name
        rows.append(agg_row)

    if not rows:
        return pd.DataFrame(columns=[c["id"] for c in spec["columns"]])
    return pd.DataFrame(rows)[[c["id"] for c in spec["columns"]]]


# ---------- Platform view ----------

def _render_metadata(spec: dict) -> None:
    st.markdown(f"### {spec.get('display_name') or spec.get('id', 'Untitled')}")
    if desc := spec.get("description"):
        st.caption(desc)

    meta = []
    if grain := spec.get("grain"):
        meta.append(f"Grain = `{grain}`")
    if vt := spec.get("viz_type"):
        meta.append(f"Viz = `{vt}`")
    if sv := spec.get("schema_version"):
        meta.append(f"schema_version `{sv}`")
    if meta:
        st.caption(" · ".join(meta))


def render_platform() -> None:
    st.header("Platform — render a spec")
    st.caption(
        f"Consumer side of the loop. Reads `{CONFIG_PATH.name}` from disk — "
        "edits saved in the Feature view show up here."
    )

    try:
        spec = load_config()
    except FileNotFoundError:
        st.error(f"No spec at {CONFIG_PATH}.")
        return

    _render_metadata(spec)

    spec_columns = spec.get("columns") or []
    if not spec_columns:
        st.warning("Spec has no columns. Add some in the Feature view.")
        return

    try:
        df = load_dataframe(spec["source"])
    except (FileNotFoundError, KeyError) as e:
        st.error(f"Could not load data source: {e}")
        return

    schema_version = spec.get("schema_version", 1)
    if schema_version >= 2 and "period" in spec:
        _render_v2(spec, df)
    else:
        _render_v1(spec, df)

    with st.expander("Active spec (raw)"):
        st.code(yaml.safe_dump(spec, sort_keys=False), language="yaml")


def _render_v1(spec: dict, df: pd.DataFrame) -> None:
    spec_columns = spec.get("columns") or []
    spec_col_ids = [col["id"] for col in spec_columns]
    present = [c for c in spec_col_ids if c in df.columns]
    missing = [c for c in spec_col_ids if c not in df.columns]
    if missing:
        st.warning(
            "Columns referenced by the spec but not in the data: "
            + ", ".join(f"`{c}`" for c in missing)
            + ". Rename them to match the CSV, or update the data source."
        )
    if not present:
        st.error("No spec columns match data columns. Check column IDs against the CSV.")
        return

    sort_filtered = [s for s in (spec.get("sort") or []) if s["column"] in df.columns]
    df = apply_sort(df, sort_filtered)

    column_config = {
        col["id"]: st.column_config.Column(label=col["display_name"])
        for col in spec_columns
        if col["id"] in df.columns
    }

    st.dataframe(
        df[present],
        column_config=column_config,
        hide_index=True,
        width="stretch",
    )


def _render_v2(spec: dict, df: pd.DataFrame) -> None:
    period_cfg = spec["period"]
    field = period_cfg["field"]
    if field not in df.columns:
        st.error(f"Period field `{field}` not in the data source.")
        return

    options = period_cfg.get("options") or [period_cfg.get("default_days", 30)]
    default = period_cfg.get("default_days", options[0])
    default_idx = options.index(default) if default in options else 0

    days = st.radio(
        "Period",
        options,
        index=default_idx,
        horizontal=True,
        format_func=lambda d: f"Last {d} days",
        key="platform_period_days",
    )

    # As-of disclosure: trend compares this window vs the immediately prior one.
    df_dates = pd.to_datetime(df[field])
    latest = df_dates.max()
    window_start = (latest - pd.Timedelta(days=days)).date()
    st.caption(
        f"As of {latest.date().isoformat()} · "
        f"window {window_start.isoformat()} → {latest.date().isoformat()} · "
        f"trend vs. prior {days}d"
    )

    try:
        agg_df = aggregate_v2(spec, df, days)
    except (KeyError, ValueError) as e:
        st.error(f"Aggregation failed: {e}")
        return

    if agg_df.empty:
        st.info("No rows in the selected window.")
        return

    sort_filtered = [s for s in (spec.get("sort") or []) if s["column"] in agg_df.columns]
    agg_df = apply_sort(agg_df, sort_filtered)

    display_df = agg_df.copy()
    for col in spec["columns"]:
        cid = col["id"]
        if cid in display_df.columns:
            ctype = col.get("type", "string")
            display_df[cid] = display_df[cid].apply(lambda v, ct=ctype: _format_value(v, ct))

    column_config = {
        col["id"]: st.column_config.Column(label=col["display_name"])
        for col in spec["columns"]
        if col["id"] in display_df.columns
    }

    st.dataframe(
        display_df,
        column_config=column_config,
        hide_index=True,
        width="stretch",
    )


# ---------- Feature view ----------

def seed_columns(spec: dict) -> pd.DataFrame:
    cols = spec.get("columns") or []
    if not cols:
        return pd.DataFrame([{"id": "", "display_name": "", "type": "string"}])
    return pd.DataFrame(cols)[["id", "display_name", "type"]]


def seed_sort(spec: dict) -> pd.DataFrame:
    sort = spec.get("sort") or []
    if not sort:
        return pd.DataFrame(columns=["column", "direction"])
    return pd.DataFrame(sort)[["column", "direction"]]


def build_spec(
    *,
    spec_id: str,
    display_name: str,
    description: str,
    grain: str,
    viz_type: str,
    source_path: str,
    columns_df: pd.DataFrame,
    sort_df: pd.DataFrame,
) -> dict:
    columns: list[dict] = []
    for _, row in columns_df.iterrows():
        col_id = (row.get("id") or "").strip()
        if not col_id:
            continue
        columns.append(
            {
                "id": col_id,
                "display_name": (row.get("display_name") or col_id).strip(),
                "type": row.get("type") or "string",
            }
        )

    sort: list[dict] = []
    for _, row in sort_df.iterrows():
        col = (row.get("column") or "").strip()
        if not col:
            continue
        sort.append({"column": col, "direction": row.get("direction") or "desc"})

    spec: dict = {
        "schema_version": FEATURE_AUTHORING_SCHEMA_VERSION,
        "viz_type": viz_type or "table",
    }
    spec["id"] = (spec_id or "spec").strip()
    if display_name:
        spec["display_name"] = display_name
    if description:
        spec["description"] = description
    if grain:
        spec["grain"] = grain
    spec["source"] = {"type": "csv", "path": source_path}
    spec["columns"] = columns
    if sort:
        spec["sort"] = sort

    return spec


def render_feature() -> None:
    st.header("Feature — author a spec")
    st.caption(
        f"Producer side of the loop. Edits are saved to `{CONFIG_PATH.name}` "
        "automatically. The Platform view renders the latest saved spec."
    )

    # Cache the on-disk spec as a stable seed for the whole session.
    # Stable seed = data_editor's cached edits track correctly; auto-write
    # doesn't shift the base out from under user edits.
    if "_feature_seed" not in st.session_state:
        try:
            st.session_state["_feature_seed"] = load_config()
        except FileNotFoundError:
            st.session_state["_feature_seed"] = {}
    seed = st.session_state["_feature_seed"]

    seed_version = seed.get("schema_version", 1) if seed else 1
    if seed_version > FEATURE_AUTHORING_SCHEMA_VERSION:
        st.warning(
            f"This spec is `schema_version: {seed_version}` "
            f"(period filters, per-column aggregation). The Feature view "
            f"only authors v{FEATURE_AUTHORING_SCHEMA_VERSION} specs today, "
            "so it's read-only here to avoid losing aggregation metadata. "
            "Edit the YAML directly for now."
        )
        st.code(yaml.safe_dump(seed, sort_keys=False), language="yaml")
        return

    # Identity
    col_a, col_b = st.columns([1, 2])
    with col_a:
        spec_id = st.text_input(
            "ID (slug)",
            value=seed.get("id", "advisor_attention"),
            help="snake_case identifier",
            key="w_spec_id",
        )
    with col_b:
        display_name = st.text_input(
            "Display name",
            value=seed.get("display_name", ""),
            key="w_display_name",
        )

    description = st.text_area(
        "Description",
        value=seed.get("description", ""),
        height=70,
        key="w_description",
    )

    # Grain + viz type
    col_grain, col_viz = st.columns(2)
    with col_grain:
        seed_grain = seed.get("grain", "service_advisor")
        seed_in_options = seed_grain in GRAIN_OPTIONS
        default_index = (
            GRAIN_OPTIONS.index(seed_grain)
            if seed_in_options
            else GRAIN_OPTIONS.index("(custom)")
        )
        grain_choice = st.selectbox(
            "Grain (row entity)",
            GRAIN_OPTIONS,
            index=default_index,
            help="What does each row represent?",
            key="w_grain_choice",
        )
        if grain_choice == "(custom)":
            custom_default = seed_grain if seed_grain and not seed_in_options else ""
            grain = st.text_input(
                "Custom grain",
                value=custom_default,
                key="w_grain_custom",
            )
        else:
            grain = grain_choice

    with col_viz:
        viz_type = st.radio(
            "Visualization type",
            VIZ_TYPES,
            index=0,
            horizontal=True,
            help="Today: table. Chart / KPI / etc. come later.",
            key="w_viz_type",
        )

    # Source
    source_default = seed.get("source", {}).get(
        "path", f"fixtures/{spec_id or 'spec'}.csv"
    )
    source_path = st.text_input(
        "Data source CSV (path relative to POC root)",
        value=source_default,
        help="The renderer reads this file at runtime.",
        key="w_source_path",
    )

    # Columns
    st.markdown("**Columns** — what each row shows, in display order")
    columns_df = st.data_editor(
        seed_columns(seed),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "id": st.column_config.TextColumn("ID", help="snake_case", required=True),
            "display_name": st.column_config.TextColumn(
                "Display name", required=True
            ),
            "type": st.column_config.SelectboxColumn(
                "Type", options=COLUMN_TYPES, required=True
            ),
        },
        key="columns_editor",
    )

    # Sort
    st.markdown("**Sort** — applied top-to-bottom")
    valid_col_ids = [c for c in columns_df["id"].dropna().tolist() if c]
    sort_df = st.data_editor(
        seed_sort(seed),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "column": st.column_config.SelectboxColumn(
                "Column", options=valid_col_ids, required=True
            ),
            "direction": st.column_config.SelectboxColumn(
                "Direction",
                options=["asc", "desc"],
                default="desc",
                required=True,
            ),
        },
        key="sort_editor",
    )

    # Build the spec from current widget state
    spec = build_spec(
        spec_id=spec_id,
        display_name=display_name,
        description=description,
        grain=grain,
        viz_type=viz_type,
        source_path=source_path,
        columns_df=columns_df,
        sort_df=sort_df,
    )

    # Persist to disk if it changed
    wrote = save_config_if_changed(spec)

    st.divider()
    save_caption = f"Saved to `{CONFIG_PATH.name}`" if wrote else f"In sync with `{CONFIG_PATH.name}`"
    st.markdown(f"**Generated spec** — {save_caption}")
    yaml_str = yaml.safe_dump(spec, sort_keys=False)
    fmt_yaml, fmt_json = st.tabs(["YAML", "JSON"])
    with fmt_yaml:
        st.code(yaml_str, language="yaml")
    with fmt_json:
        st.code(json.dumps(spec, indent=2), language="json")

    st.download_button(
        label=f"Download {spec_id or 'spec'}.yaml",
        data=yaml_str,
        file_name=f"{spec_id or 'spec'}.yaml",
        mime="text/yaml",
    )


# ---------- Main ----------

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("# Table View POC")
        st.caption("Author and render report-definition specs.")
        page = st.radio(
            "View",
            ["Platform", "Feature"],
            captions=["Render a spec", "Author a spec"],
            label_visibility="collapsed",
        )
    return page


def main() -> None:
    st.set_page_config(page_title="Table View POC", layout="wide")
    page = render_sidebar()
    if page == "Platform":
        render_platform()
    else:
        render_feature()


if __name__ == "__main__":
    main()
