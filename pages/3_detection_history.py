"""Local detection-history table, overview chart, and export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.storage import list_detection_records, records_to_csv
from utils.ui import page_intro, section_heading

page_intro(
    "03",
    "inspection archive",
    "A memory for every scan.",
    "Review saved summaries, compare inspection volume, and export the record when you need it.",
)

records = list_detection_records()
if not records:
    st.info(
        "No saved records yet. Run a detection and choose Save summary to history.",
        icon=":material/history:",
    )
    st.stop()

data = pd.DataFrame(records)
section_heading("Focus the record", "Filter saved summaries by their source media.")
selected_types = st.segmented_control(
    "Show input types",
    options=sorted(data["input_type"].dropna().unique()),
    default=sorted(data["input_type"].dropna().unique()),
    selection_mode="multi",
    width="stretch",
)
filtered = data[data["input_type"].isin(selected_types or [])]

section_heading(
    "Read the inspection history",
    "Review saved results and export the visible records when needed.",
)
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Saved records", len(filtered))
metric_b.metric("Total detected potholes", int(filtered["detection_count"].sum()))
average_ms = filtered["processing_ms"].dropna().mean()
metric_c.metric(
    "Average processing time", f"{average_ms:.0f} ms" if pd.notna(average_ms) else "—"
)

display_columns = [
    "created_at",
    "input_type",
    "input_name",
    "detection_count",
    "average_confidence",
    "apparent_severity",
    "processing_ms",
]
with st.container(border=True):
    st.dataframe(
        filtered[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "created_at": st.column_config.DatetimeColumn("Saved at"),
            "input_type": st.column_config.TextColumn("Media"),
            "input_name": st.column_config.TextColumn("Source"),
            "detection_count": st.column_config.NumberColumn("Potholes", format="%d"),
            "average_confidence": st.column_config.NumberColumn(
                "Confidence", format="percent"
            ),
            "apparent_severity": st.column_config.TextColumn("Apparent severity"),
            "processing_ms": st.column_config.NumberColumn(
                "Processing", format="%.0f ms"
            ),
        },
    )

if not filtered.empty:
    chart_data = filtered.groupby("input_type", as_index=False)["detection_count"].sum()
    st.bar_chart(
        chart_data,
        x="input_type",
        y="detection_count",
        x_label="Media type",
        y_label="Potholes detected",
    )

with st.container(horizontal=True, horizontal_alignment="right"):
    st.download_button(
        "Export archive",
        data=records_to_csv(filtered.to_dict(orient="records")),
        file_name="pothole_detection_history.csv",
        mime="text/csv",
        icon=":material/download:",
    )
