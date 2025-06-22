from fastapi import FastAPI,Request,UploadFile,File,Form,HTTPException
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from app.cleaning import quick_clean, thorough_clean
from app.utils import generate_all_charts, generate_timeline_chart,generate_correlation_heatmap
from app.data_cleaning import get_cleaning_suggestions
from typing import List

app = FastAPI()

app.mount("/static",StaticFiles(directory="app/static"),name="static")
templates = Jinja2Templates(directory= "app/templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR,exist_ok= True)


@app.get("/",response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("upload_form.html",{"request":request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    chart_type: str = Form("bar")
):
    # Clean up old files
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))

    csv_path = os.path.join(UPLOAD_DIR, "latest.csv")
    with open(csv_path, "wb+") as f:
        f.write(await file.read())

    df = pd.read_csv(csv_path)
    df.to_pickle(os.path.join(UPLOAD_DIR, "latest.pkl"))

    eligible_columns = [
        col for col in df.columns
        if df[col].dtype == "object" or df[col].nunique() < 30
    ]

    return templates.TemplateResponse("select_columns.html", {
        "request": request,
        "filename": file.filename,
        "columns": eligible_columns,
        "chart_type": chart_type
    })
    
@app.post("/visualize", response_class=HTMLResponse)
async def visualize_selected_columns(
    request: Request,
    selected_columns: List[str] = Form(...),
    chart_type: str = Form("bar"),
):
    df = pd.read_pickle(os.path.join(UPLOAD_DIR, "latest.pkl"))

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist()
    }

    chart_html_list = generate_all_charts(df, chart_type, selected_columns)
    timeline_html = generate_timeline_chart(df)
    cleaning_suggestions = get_cleaning_suggestions(df)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "summary": summary,
        "filename": "latest.csv",
        "charts": chart_html_list,
        "chart_type": chart_type,
        "timeline_chart": timeline_html,
        "cleaning": cleaning_suggestions
    })
    
@app.post("/filter", response_class=HTMLResponse)
async def filter_data(
    request: Request,
    selected_columns: List[str] = Form(...),
    chart_type: str = Form("bar")
):
    df = pd.read_pickle(os.path.join(UPLOAD_DIR, "latest.pkl"))

    # Split columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= 20]
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    return templates.TemplateResponse("filter_form.html", {
        "request": request,
        "selected_columns": selected_columns,
        "chart_type": chart_type,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    })


@app.post("/dashboard", response_class=HTMLResponse)
async def filtered_dashboard(
    request: Request,
    selected_columns: List[str] = Form(...),
    chart_type: str = Form("bar"),
    cat_filter_col: str = Form(None),
    cat_filter_val: str = Form(None),
    num_filter_col: str = Form(None),
    num_min: float = Form(None),
    num_max: float = Form(None),
):
    df = pd.read_pickle(os.path.join(UPLOAD_DIR, "latest.pkl"))

    # Apply categorical filter
    if cat_filter_col and cat_filter_val:
        df = df[df[cat_filter_col].astype(str) == str(cat_filter_val)]

    # Apply numeric filter
    if num_filter_col and num_min is not None and num_max is not None:
        df = df[(df[num_filter_col] >= num_min) & (df[num_filter_col] <= num_max)]

    chart_html_list = generate_all_charts(df, chart_type, selected_columns)
    timeline_html = generate_timeline_chart(df)
    cleaning_suggestions = get_cleaning_suggestions(df)
    correlation_html = generate_correlation_heatmap(df)

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist()
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "summary": summary,
        "filename": "filtered.csv",
        "charts": chart_html_list,
        "chart_type": chart_type,
        "timeline_chart": timeline_html,
        "correlation_heatmap": correlation_html,
        "cleaning": cleaning_suggestions
    })

    


@app.post("/deepclean", response_class=HTMLResponse)
async def deep_clean_data(request: Request, filename: str = Form(...), mode: str = Form("quick")):
    file_path = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(file_path)
    original_shape = df.shape
    original_cols = df.columns.tolist()

    if mode == "quick":
        df_clean = quick_clean(df)
    else:
        df_clean = thorough_clean(df)

    cleaned_filename = f"cleaned_{mode}_{filename}"
    cleaned_path = os.path.join(UPLOAD_DIR, cleaned_filename)
    df_clean.to_csv(cleaned_path, index=False)

    # Summary changes
    change_summary = {
        "original_shape": original_shape,
        "new_shape": df_clean.shape,
        "dropped_columns": list(set(original_cols) - set(df_clean.columns)),
        "new_columns": list(set(df_clean.columns) - set(original_cols)),
    }

    chart_html_list = generate_all_charts(df_clean)
    timeline_html = generate_timeline_chart(df_clean)
    cleaning_suggestions = get_cleaning_suggestions(df_clean)

    return templates.TemplateResponse("dashboard_cleaned.html", {
        "request": request,
        "filename": cleaned_filename,
        "summary": {
            "num_rows": len(df_clean),
            "num_columns": len(df_clean.columns),
            "columns": df_clean.columns.tolist()
        },
        "charts": chart_html_list,
        "timeline_chart": timeline_html,
        "cleaning": cleaning_suggestions,
        "changes": change_summary
    })

@app.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(path=filepath, filename=filename, media_type='text/csv')

from fastapi.responses import FileResponse
from ydata_profiling import ProfileReport
import os
import pandas as pd

@app.get("/profiling-report", response_class=HTMLResponse)
async def profiling_report():
    import os
    from ydata_profiling import ProfileReport
    import pandas as pd

    try:
        df = pd.read_pickle("uploads/latest.pkl")
        profile = ProfileReport(df, title="AutoSight Data Profiling Report", explorative=True)
        html_str = profile.to_html()
        return HTMLResponse(content=html_str, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)