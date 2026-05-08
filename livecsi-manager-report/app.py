"""Streamlit app for the table-view POC.

Sidebar nav:
  - Platform: render a spec (consumer side of the loop)
  - Feature:  author a spec interactively (producer side of the loop)

Spec field naming follows numa-metrics-registry conventions where they fit:
  schema_version, id, display_name, description, grain. The registry's
  `dimensions` term means pivot axes (e.g. rooftop_id) — distinct from the
  output `columns` of a table view, so we keep `columns` here.
"""

from __future__ import annotations

import json
from pathlib import Path

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
SCHEMA_VERSION = 1


# ---------- Loaders ----------

def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_dataframe(source: dict) -> pd.DataFrame:
    return pd.read_csv(POC_ROOT / source["path"])


def apply_sort(df: pd.DataFrame, sort_spec: list[dict] | None) -> pd.DataFrame:
    if not sort_spec:
        return df
    by = [s["column"] for s in sort_spec]
    ascending = [s.get("direction", "asc") == "asc" for s in sort_spec]
    return df.sort_values(by=by, ascending=ascending)


# ---------- Platform view ----------

def render_platform() -> None:
    st.header("Platform — render a spec")
    st.caption(
        "Consumer side of the loop. Loads `config/advisor_attention.yaml`, "
        "applies sort, renders the table — what the in-app reporting platform "
        "will eventually do."
    )

    try:
        spec = load_config()
    except FileNotFoundError:
        st.error(f"No spec at {CONFIG_PATH}")
        return

    st.markdown(f"### {spec.get('display_name', spec['id'])}")
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

    df = load_dataframe(spec["source"])
    df = apply_sort(df, spec.get("sort"))

    columns_in_order = [col["id"] for col in spec["columns"]]
    column_config = {
        col["id"]: st.column_config.Column(label=col["display_name"])
        for col in spec["columns"]
    }

    st.dataframe(
        df[columns_in_order],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Loaded spec (raw)"):
        st.code(yaml.safe_dump(spec, sort_keys=False), language="yaml")


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

    spec: dict = {"schema_version": SCHEMA_VERSION, "viz_type": viz_type or "table"}
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
        "Producer side of the loop. Pick the row entity, columns, sort, viz type — "
        "the spec is generated live below. Download the YAML and drop it in `config/` to "
        "render via the Platform view."
    )

    try:
        seed = load_config()
    except FileNotFoundError:
        seed = {}

    # Identity
    col_a, col_b = st.columns([1, 2])
    with col_a:
        spec_id = st.text_input(
            "ID (slug)",
            value=seed.get("id", "advisor_attention"),
            help="snake_case identifier",
        )
    with col_b:
        display_name = st.text_input(
            "Display name", value=seed.get("display_name", "")
        )

    description = st.text_area(
        "Description", value=seed.get("description", ""), height=70
    )

    # Grain + viz type
    col_grain, col_viz = st.columns(2)
    with col_grain:
        seed_grain = seed.get("grain", "service_advisor")
        if seed_grain in GRAIN_OPTIONS:
            choice = st.selectbox(
                "Grain (row entity)",
                GRAIN_OPTIONS,
                index=GRAIN_OPTIONS.index(seed_grain),
                help="What does each row represent? (registry term: `grain`)",
            )
            grain = (
                st.text_input("Custom grain", value="") if choice == "(custom)" else choice
            )
        else:
            st.selectbox(
                "Grain (row entity)",
                GRAIN_OPTIONS,
                index=GRAIN_OPTIONS.index("(custom)"),
            )
            grain = st.text_input("Custom grain", value=seed_grain)

    with col_viz:
        viz_type = st.radio(
            "Visualization type",
            VIZ_TYPES,
            index=0,
            horizontal=True,
            help="Today: table. Chart / KPI / etc. come later.",
        )

    # Source
    source_default = seed.get("source", {}).get(
        "path", f"fixtures/{spec_id or 'spec'}.csv"
    )
    source_path = st.text_input(
        "Data source CSV (path relative to POC root)",
        value=source_default,
        help="The renderer reads this file at runtime.",
    )

    # Columns
    st.markdown("**Columns** — what each row shows, in display order")
    columns_df = st.data_editor(
        seed_columns(seed),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
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
        use_container_width=True,
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

    # Build + preview
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

    st.divider()
    st.markdown("**Generated spec** (live)")
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
        st.markdown("# LiveCSI Manager Report POC")
        st.caption("Author and render report-definition specs.")
        page = st.radio(
            "View",
            ["Platform", "Feature"],
            captions=["Render a spec", "Author a spec"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Field naming follows numa-metrics-registry conventions "
            "(`schema_version`, `id`, `display_name`, `grain`) where they fit. "
            "Scaled-back POC — does not model the full registry shape "
            "(`domain`, `owner_team`, `lifecycle`, `lineage`, etc.)."
        )
    return page


def main() -> None:
    st.set_page_config(page_title="LiveCSI Manager Report POC", layout="wide")
    page = render_sidebar()
    if page == "Platform":
        render_platform()
    else:
        render_feature()


if __name__ == "__main__":
    main()
