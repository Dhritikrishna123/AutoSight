import pandas as pd
import plotly.express as px
import warnings
from typing import Optional,List

def generate_all_charts(df: pd.DataFrame, chart_type: str = "bar", selected_columns: Optional[List[str]] = None) -> list:
    import plotly.express as px

    charts = []
    columns = selected_columns or df.columns

    for col in columns:
        if chart_type in ["bar", "pie"]:
            if df[col].dtype == 'object' or df[col].nunique() < 30:
                top_values = df[col].value_counts().nlargest(10)
                if chart_type == "bar":
                    fig = px.bar(x=top_values.index, y=top_values.values, title=f"Top {col}")
                elif chart_type == "pie":
                    fig = px.pie(names=top_values.index, values=top_values.values, title=f"{col} Distribution")
                charts.append(fig.to_html(full_html=False))

        elif chart_type == "line":
            if pd.api.types.is_numeric_dtype(df[col]):
                fig = px.line(df, y=col, title=f"{col} Line Chart")
                charts.append(fig.to_html(full_html=False))

        elif chart_type == "scatter":
            if pd.api.types.is_numeric_dtype(df[col]):
                # Pick another numeric column as x (not equal to y)
                numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != col]
                if numeric_cols:
                    fig = px.scatter(df, x=numeric_cols[0], y=col, title=f"{col} vs {numeric_cols[0]}")
                    charts.append(fig.to_html(full_html=False))

    return charts






def generate_timeline_chart(df: pd.DataFrame) -> str:
    date_keywords = ["date", "time", "created", "posted", "timestamp", "datetime"]

    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[col], errors="coerce")

                if parsed.notna().sum() < 3:
                    continue  # skip if <3 valid dates

                count_by_date = parsed.dt.date.value_counts().sort_index()
                fig = px.line(
                    x=count_by_date.index,
                    y=count_by_date.values,
                    labels={'x': 'Date', 'y': 'Count'},
                    title=f"Timeline: {col}"
                )
                return fig.to_html(full_html=False)

            except Exception:
                continue

    return ""  # No usable date/time column found

def generate_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return ""  # Not enough data for correlation

    corr_matrix = numeric_df.corr().round(2)
    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="📊 Correlation Matrix")
    return fig.to_html(full_html=False)


