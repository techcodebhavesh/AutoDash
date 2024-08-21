import pandas as pd
import os
import json
import re
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe
from dotenv import load_dotenv
import requests
from app.prompts.GenerateBar import  GenerateBar
from app.prompts.GeneratePie import  GeneratePie
from app.prompts.GenerateLineSingle import GenerateLineSingle
from app.prompts.GenerateLineMultiple import  GenerateLineMultiple
from app.prompts.GenerateScatter import  GenerateScatterPlot

from app.prompts.Howmany import  Howmany
from string import Template

# Load environment variables
load_dotenv()

# Set directory name
dirname = os.path.dirname(__file__)

# generate_bar = GenerateBar()
# Format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format
res = None  # Initialize res as None
resdata=None

# Read CSV file into a DataFrame
print('genhelo')
def data_to_flask(req_data):
    df = pd.read_csv(req_data["file"])
    # Define dfs as a dictionary of DataFrames
    dfs = {
        0: df  # Use df read from the CSV
    }

    # Initialize LLM
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        api_key=os.getenv("API_KEY")
    )

    def convert_df_to_csv(df: pd.DataFrame, extras: dict) -> str:
        dataframe_info = "<dataframe"

        # Add name attribute if available
        if 'name' in extras:
            dataframe_info += f' name="{extras["name"]}"'

        # Add description attribute if available
        if 'description' in extras:
            dataframe_info += f' description="{extras["description"]}"'

        dataframe_info += ">"

        # Add DataFrame details
        dataframe_info += f"\ndfs[{extras['index']}]:{df.shape[0]}x{df.shape[1]}\n{df.head(8).to_csv(index=False)}"

        # Close the DataFrame tag
        dataframe_info += "</dataframe>\n"

        return dataframe_info
    
    def setDescriptionToTemplate(templateData, description):
        template = Template(templateData)
        return template.substitute(description=description)

    # Generate a small description of the DataFrame
    description = convert_df_to_csv(df, {"index": 0})
    print("*****282373***")
    print(description)
    
    # json_data = f"""{Howmany()}"""    
    
    # print(json_data)

    # print(type(json_data))
    # print(json_data)
    # howmany= json.dumps(json_data)
#     props = {
#     "context": "provide a matplotlib graph code in the  python code everytime" ,  # provide actual context
#     "code": "qo" ,     # provide actual code
#     "error": "heu" ,    # provide actual error
# }
    howmany=Howmany()
    print(type(howmany))
    string_r = f"""{howmany.to_string()}"""
    print("string_r")
    print(string_r)
    string_r = setDescriptionToTemplate(string_r, description)
    print("string_r")
    print(string_r)
    
    




    print(string_r)
    print(type(string_r))
    res = llm.invoke(string_r)
    print(res)
    
    resdata=res.content
    # print(resdata)
    
    resdata_dict = json.loads(resdata)
    # print(type(resdata_dict))
    print(resdata_dict)
    



    
    


    # Define prompts for each chart type
    prompts = {
        "bar_chart": f"""{GenerateBar()}""",
        "pie_chart":f"""{GeneratePie()}""",
        "line_chart_single": f"""{GenerateLineSingle()}""",
        "line_chart_multiple": f"""{GenerateLineMultiple()}""",
        "scatter_plot":f"""{GenerateScatterPlot()}""",
    }
    
    for chart_type, prompt in prompts.items():
        prompts[chart_type] = setDescriptionToTemplate(prompt, description)
        print(setDescriptionToTemplate(prompt, description))
   


    def sanitize_quotes(code_str):
        # Replace curly quotes with straight quotes
        return (code_str
                .replace('“', '"')
                .replace('”', '"')
                .replace('‘', "'")
                .replace('’', "'")
                .replace('\u201c', '"')  # Handle specific Unicode characters
                .replace('\u201d', '"'))

    def remove_non_printable(code_str):
        # Remove non-printable characters
        return re.sub(r'[^\x00-\x7F]+', '', code_str)


    # Function to query LLM and extract code
    def get_python_code_for_prompt(prompt):
        response = llm.invoke(prompt)
        response_content = response.content
        print(response_content)

        # Find the indices of code block delimiters
        start_index = response_content.find("```Python") + 9
        if start_index == 8:  # Adjusting for cases where `Python` keyword might be missing
            start_index = response_content.find("```python") + 9
        if start_index == 8:  # Fallback for generic code blocks
            start_index = response_content.find("```") + 3
        
        if start_index == 2:
            return None
        
        end_index = response_content.find("```", start_index)

        python_code = response_content[start_index:end_index].strip()
        
        # Sanitize the code
        python_code = sanitize_quotes(python_code)
        python_code = remove_non_printable(python_code)
        print(python_code)
        return python_code

    # Function to execute code and format output
    def execute_and_format_code(code_str, context):
        try:
            # Add dirname and dfs to context
            exec_context = context.copy()
            exec_context['dirname'] = dirname  # Include dirname in the context
            exec_context.update(dfs)  # Include dfs in the context

            # Adjust code to use the correct file path
            code_str = code_str.replace('path_to_your_csv_file.csv', req_data["file"])

            print("------------------------------------------------------------------------")
            print(code_str)
            print("------------------------------------------------------------------------")

            # Execute the user code
            result = exec(code_str, exec_context)
            
            # Retrieve the result from the execution context
            result = exec_context.get('results', [])
            
            # Return the result and the code used for debugging
            return {
                "code": code_str,
                "result": result
            }
        except Exception as e:
            # Return the code and the error message
            return {
                "code": code_str,
                "error": str(e)
            }

    # Prepare the execution context
    execution_context = {"pd": pd, "df": df, **dfs}
    # json_data = json.loads(res["content"])
    # print("#################################################")
    # print(json_data)
    # Query LLM with each prompt and extract code
    extracted_code = {}
    print(prompts)
    
    for chart_type, prompt in prompts.items():
        # print(res)
        # print(chart_type)
        
        # print(resdata)
        print(type(resdata))
        # print(resdata[chart_type])
        print(resdata_dict[chart_type])

        if   resdata_dict[chart_type]:
            print(f"Querying LLM for {chart_type}...")
            code =  get_python_code_for_prompt(prompt)
            print(code)
            if code:
                extracted_code[chart_type] = code

    # Execute the extracted code and format the output
    results = []
    for chart_type, code in extracted_code.items():
        print(f"Executing code for {chart_type}...")
        result = execute_and_format_code(code, execution_context)
        
        # Format the results for frontend
        if 'error' in result:
            output = {
                "chartType": chart_type.upper().replace("_", " "),
                "chartData": {
                    "error": result['error'],
                    "code": result['code']
                }
            }
        else:
            output = {
                "chartType": chart_type.upper().replace("_", " "),
                "chartData": result['result'] if result['result'] else None,
                "code": result['code']  # Include the code used for debugging
            }
        
        results.append(output)

    # Print final results
    print("testinggggggggg")
    # print(json.dumps(results, indent=4))
    
    

    return results
