import pandas as pd
from ragdash.chromadb import ChromaDB_VectorStore
from ragdash.groq import Groq
from ragdash.exceptions import ValidationError  
from dotenv import load_dotenv
import os
import gradio as gr
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"
load_dotenv()

class MyRAGdash(ChromaDB_VectorStore, Groq):
    def __init__(self, config=None):
        # Initialize both base classes
        ChromaDB_VectorStore.__init__(self, config=config)
        Groq.__init__(self, config=config)

# Usage example
config = {
    'db_user': os.getenv('DB_USER'),
    'db_password': os.getenv('DB_PASSWORD'),
    'db_host': os.getenv('DB_HOST'),
    'db_name': os.getenv('DB_NAME')
}

print(config)

rd = MyRAGdash(config=config)

def generate_charts(df):
    # Define the types of charts
    my_questions = {
        "bar": "extract the data appropriate for bar chart",
        "pie": "extract the data appropriate for pie chart",
        "line": "extract the data appropriate for line chart",
        "scatter": "extract the data appropriate for scatter chart",
        "bubble": "extract the data appropriate for bubble chart",
        "heatmap": "extract the data appropriate for heatmap chart",
        "box": "extract the data appropriate for box chart",
        "histogram": "extract the data appropriate for histogram chart",
    }

    results = {}
    charts = []

    for chart_type, question in my_questions.items():
        try:
            python_code = rd.generate_python(df=df, type_c=chart_type)
            extracted_df = rd.extract_dataframe(python_code, df=df)

            if isinstance(extracted_df, str):
                charts.append((f"{chart_type.capitalize()} Chart Error", extracted_df))
                continue

            # Generate Plotly code and figure
            code = rd.generate_plotly_code(question=question, df=extracted_df, df_metadata=extracted_df.columns, type_c=chart_type)
            fig = rd.get_plotly_figure(plotly_code=code, df=extracted_df)

            # Add Plotly figure to the list of charts
            charts.append((chart_type.capitalize(), fig))

        except ValidationError as e:
            charts.append((f"{chart_type.capitalize()} Chart Error", f"Validation error for {chart_type} chart: {e}"))
        except Exception as e:
            charts.append((f"{chart_type.capitalize()} Chart Error", f"Error processing {chart_type} chart: {e}"))

    return charts

# Gradio Interface
def dashboard(csv_file):
    if csv_file is None:
        return []

    # Read the uploaded CSV file
    df_i = pd.read_csv(csv_file.name)

    # Generate charts based on the uploaded CSV
    charts = generate_charts(df_i)

    # Create chart elements using Gradio's Row and Column for layout
    chart_elements = []
    i = 0
    while i < len(charts):
        with gr.Row():
            i_row = i + 3
            while i < i_row:
                if i < len(charts):
                    title, fig = charts[i]
                    if isinstance(fig, str):
                        # Display error message
                        with gr.Column():
                            gr.Markdown(f"### {title}")
                            gr.Markdown(fig)
                    else:
                        # Display the Plotly chart
                        with gr.Column():
                            gr.Markdown(f"### {title}")
                            gr.Plot(fig)
                i += 1

    return chart_elements

# Define Gradio layout with file upload and dynamic dashboard display
with gr.Blocks() as app:
    gr.Markdown("# Data Visualization Dashboard")
    
    # File uploader for CSV files
    csv_file = gr.File(label="Upload CSV File", file_types=["csv"])
    
    # Dashboard output section
    dashboard_output = gr.Row().update(dashboard, csv_file)

app.launch(share=True)
