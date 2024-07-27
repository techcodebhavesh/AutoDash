# Assume openai>=1.0.0
from openai import OpenAI
import json
from datetime import datetime
import time as t

# Create an OpenAI client with your KRUTRIM API KEY and endpoint
openai = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1",
)


# Function to validate JSON
# def validate_json(json_str):
#     try:
#         json_obj = json.loads(json_str)
#         return json_obj
#     except json.JSONDecodeError:
#         return None
# Multiline prompt

data_types = ["Sales Revenue", "Profit Margins", "Cost of Goods Sold (COGS)", "Net Income", "Employee Salaries", "Customer Demographics", "Market Share", "Product Prices", "Inventory Levels", "Operating Expenses", "Advertising Spend", "Customer Satisfaction Scores", "Website Traffic", "Conversion Rates", "Lead Generation", "Customer Retention Rates", "Sales by Region", "Product Returns", "Shipping Costs", "Vendor Payments", "Tax Payments", "Capital Expenditures", "Utility Costs", "Depreciation and Amortization", "Cash Flow", "Debt Levels", "Equity Holdings", "Dividends Paid", "Stock Prices", "R&D Expenses", "Patents Held", "Market Research Data", "Competitor Analysis", "Supply Chain Metrics", "Production Volume", "Quality Control Metrics", "Employee Turnover Rates", "Training Expenses", "Health and Safety Incidents", "Customer Service Interactions", "Loyalty Program Data", "E-commerce Sales", "Retail Sales", "Wholesale Sales", "B2B Sales", "B2C Sales", "Partnership Agreements", "Franchise Operations", "Sponsorship Deals", "Event Participation"]

multiline_prompt = """
Here are some examples of interactions:
[
    {
        "instructions": "Determine the average score for each subject from the given exam results.",
        "input": "<dataframe>\\n                 dfs[0]:5x4\\n                 student_id,math,science,english\\n                 1,78,85,90\\n                 2,88,79,92\\n                 3,95,91,85\\n                 4,85,87,88\\n                 5,90,93,92\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Calculate the average score per subject\\naverage_scores = df[['math', 'science', 'english']].mean()\\n\\n# Display the average scores\\nprint(average_scores)"
    },
    {
        "instructions": "Create a trend plot for closing stock prices over time.",
        "input": "<dataframe>\\n                 dfs[0]:5x6\\n                 date,open,high,low,close,volume\\n                 2023-07-01,150,155,148,153,1000000\\n                 2023-07-02,153,158,151,157,1100000\\n                 2023-07-03,157,160,155,159,1050000\\n                 2023-07-04,159,162,158,161,950000\\n                 2023-07-05,161,164,160,163,900000\\n                 </dataframe>",
        "output": "import pandas as pd\\nimport matplotlib.pyplot as plt\\n\\n# Plot the closing price trends\\ndf['date'] = pd.to_datetime(df['date'])\\nplt.plot(df['date'], df['close'])\\nplt.xlabel('Date')\\nplt.ylabel('Closing Price')\\nplt.title('Closing Price Trends')\\nplt.show()"
    },
    {
        "instructions": "Identify the top 3 best-selling products based on total revenue.",
        "input": "<dataframe>\\n                 dfs[0]:5x4\\n                 product_id,product_name,units_sold,total_revenue\\n                 101,Widget A,500,10000\\n                 102,Widget B,600,15000\\n                 103,Widget C,700,20000\\n                 104,Widget D,800,25000\\n                 105,Widget E,900,30000\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Find the top 3 products with the highest sales\\ntop_products = df.nlargest(3, 'total_revenue')[['product_name', 'total_revenue']]\\n\\n# Display the top products\\nprint(top_products)"
    },
    {
        "instructions": "Calculate the average number of likes per post from the social media data.",
        "input": "<dataframe>\\n                 dfs[0]:5x3\\n                 post_id,content,likes,comments\\n                 1,Post about AI,150,20\\n                 2,Post about ML,200,30\\n                 3,Post about Data Science,250,25\\n                 4,Post about Python,300,35\\n                 5,Post about SQL,180,15\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Calculate the average likes per post\\naverage_likes = df['likes'].mean()\\n\\n# Display the average likes\\nprint(average_likes)"
    },
    {
        "instructions": "Find the day with the highest recorded temperature from the weather data.",
        "input": "<dataframe>\\n                 dfs[0]:5x3\\n                 date,temperature,humidity\\n                 2023-07-01,85,60\\n                 2023-07-02,88,65\\n                 2023-07-03,90,70\\n                 2023-07-04,92,75\\n                 2023-07-05,89,68\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Find the day with the highest temperature\\nhottest_day = df.loc[df['temperature'].idxmax()]\\n\\n# Display the hottest day\\nprint(hottest_day)"
    },
    {
        "instructions": "Compute the average age of the patients in the medical dataset.",
        "input": "<dataframe>\\n                 dfs[0]:5x3\\n                 patient_id,age,diagnosis\\n                 1,25,Flu\\n                 2,30,Cold\\n                 3,45,COVID-19\\n                 4,35,Flu\\n                 5,50,Cold\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Calculate the average age of patients\\naverage_age = df['age'].mean()\\n\\n# Display the average age\\nprint(average_age)"
    },
    {
        "instructions": "Summarize the total sales for each product category.",
        "input": "<dataframe>\\n                 dfs[0]:5x4\\n                 transaction_id,product_category,units_sold,sales_amount\\n                 1,Electronics,5,500\\n                 2,Clothing,10,300\\n                 3,Groceries,20,200\\n                 4,Furniture,3,600\\n                 5,Toys,7,100\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Find the total sales for each product category\\ntotal_sales = df.groupby('product_category')['sales_amount'].sum().reset_index()\\n\\n# Display the total sales\\nprint(total_sales)"
    },
    {
        "instructions": "Compute the average customer rating for each product from the feedback data.",
        "input": "<dataframe>\\n                 dfs[0]:5x3\\n                 product_id,product_name,rating\\n                 1,Widget A,4.5\\n                 2,Widget B,4.0\\n                 3,Widget C,3.5\\n                 4,Widget D,5.0\\n                 5,Widget E,4.2\\n                 </dataframe>",
        "output": "import pandas as pd\\n\\n# Calculate the average rating for each product\\naverage_ratings = df.groupby('product_name')['rating'].mean().reset_index()\\n\\n# Display the average ratings\\nprint(average_ratings)"
    }
]

Please create more 2 unique such prompts and outputs based on the above interactions.

note: please dont      number the prompts and outputs.
note: please provide in json array of objects format.
note: please enclose the json array within triple backticks. Use escape character to avoid formatting issues.
```[
    {
        "instructions": "",
        "input": "",
        "output": ""
    }
    {
        "instructions": "",
        "input": "",
        "output": ""
    }
]```
"""



# Create a chat completion with multiline prompt
chat_completion = openai.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content":multiline_prompt }
    ],

    max_tokens=5000,
) 

log_file = "logger.txt"

def logger(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_line = f"{timestamp} - {text}\n"
    with open(log_file, "a") as file:
        file.write(new_line)

def extract_json(content):
    print(content)
    parts = content.split('```')
    if len(parts) > 1:
        try:
            data = json.loads(parts[1].strip())
            return data
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from response")
            return None
        except Exception as e:
            print("Error: Failed to extract JSON from response")
            print(e)
            return None
    else:
        print("Error: No JSON content found within triple backticks")
        return None

# Function to generate prompts and outputs
def generate_prompts(data_type):
    chat_completion = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate a unique set of instructions, inputs, and outputs for a {data_type} dataset."},
            {"role": "user", "content": multiline_prompt}
        ],
        max_tokens=2048,
    )
    return chat_completion.choices[0].message.content

# Loop to generate and validate JSON
all_prompts = []

for i in range(len(data_types)):
    logger(f"Generating prompts and outputs for {data_types[i]}")
    for _ in range(2):
        output = generate_prompts(data_types[i])
        
        if output is not None:
            x = None
            while x is None:
                try:
                    x = extract_json(output)
                    if x is None:
                        output = generate_prompts(data_types[i])
                except Exception as e:
                    output = generate_prompts(data_types[i])
                    x = None
            
            for j in x:
                all_prompts.append(j)

            output = None
            # exit()
        else:
            t.sleep(5)
    
    logger(f"Generated prompts and outputs for {data_types[i]}")

    # Save the output to a file
    with open('output.txt', 'w') as f:
        json.dump(all_prompts, f, indent=4)

print("Generated prompts and outputs saved to output.txt")