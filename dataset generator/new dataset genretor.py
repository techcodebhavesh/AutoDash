import os
import json
import csv
from dotenv import load_dotenv
import time
from groq import Client

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Client(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com"
)

def generate_data(prompt):
    response = client.chat.create(
        model="Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
    )
    if response and response.choices:
        try:
            # Split the response content by triple backticks
            parts = response.choices[0].message.content.split('```')
            if len(parts) > 1:
                # Parse the JSON content within the triple backticks
                data = json.loads(parts[1].strip())
                return data
            else:
                print("Error: No JSON content found within triple backticks")
                return None
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from response")
            return None
    else:
        print("Error: Failed to generate data")
        return None

# Define the prompt template
prompt_template = """Generate a unique set of instructions, inputs, and outputs for a specific data type.
The output should be a JSON object enclosed within triple backticks (```):
```
{
    "instruction": "Your instruction here",
    "input": "Your input here",
    "output": "Your output here"
}
```
Type: 
"""

# List of types for generating diverse data
data_types = ['educational', 'sales', 'medical', 'financial', 'legal']

# Initialize an empty list to store the generated data
generated_data = []

# Generate 5 unique sets for each data type
for data_type in data_types:
    for i in range(5):
        prompt = prompt_template+data_type
        result = generate_data(prompt)
        if result:
            # Extract data from the JSON response
            instruction = result.get('instruction', 'N/A')
            input_data = result.get('input', 'N/A')
            output_data = result.get('output', 'N/A')
            generated_data.append([instruction, input_data, output_data])
        time.sleep(1)  # To avoid hitting API rate limits

# Define the CSV file path
output_csv_path = 'F:/Mayur/vit/innov8ors/ollama/AutoDash/generated_dataset.csv'

# Write the generated data to the CSV file
with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header if the file is empty
    if csvfile.tell() == 0:
        csvwriter.writerow(['Instruction', 'Input', 'Output'])
    # Write the generated data
    csvwriter.writerows(generated_data)

print("Data generation completed and saved to CSV file.")