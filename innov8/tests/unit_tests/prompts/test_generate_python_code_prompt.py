"""Unit tests for the generate python code prompt class"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from pandasai import Agent
from pandasai.connectors import PandasConnector
from pandasai.ee.connectors.relations import PrimaryKey
from pandasai.helpers.dataframe_serializer import DataframeSerializerType
from pandasai.llm.fake import FakeLLM
from pandasai.prompts import GeneratePythonCodePrompt


class TestGeneratePythonCodePrompt:
    """Unit tests for the generate python code prompt class"""

    @pytest.mark.parametrize(
        "output_type,output_type_template",
        [
            *[
                (
                    None,
                    """type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "number",
                    """type (must be "number"), value must int. Example: { "type": "number", "value": 125 }""",
                ),
                (
                    "dataframe",
                    """type (must be "dataframe"), value must be pd.DataFrame or pd.Series. Example: { "type": "dataframe", "value": pd.DataFrame({...}) }""",
                ),
                (
                    "plot",
                    """type (must be "plot"), value must be string. Example: { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "string",
                    """type (must be "string"), value must be string. Example: { "type": "string", "value": f"The highest salary is {highest_salary}." }""",
                ),
            ]
        ],
    )
    def test_str_with_args(self, output_type, output_type_template):
        """Test casting of prompt to string and interpolation of context.

        Args:
            output_type (str): output type
            output_type_template (str): output type template

        Returns:
            None
        """

        os.environ["PANDASAI_API_URL"] = ""
        os.environ["PANDASAI_API_KEY"] = ""

        llm = FakeLLM()
        agent = Agent(
            PandasConnector({"original_df": pd.DataFrame({"a": [1], "b": [4]})}),
            config={"llm": llm, "dataframe_serializer": DataframeSerializerType.CSV},
        )
        prompt = GeneratePythonCodePrompt(
            context=agent.context,
            output_type=output_type,
        )

        expected_prompt_content = f"""<dataframe>
dfs[0]:1x2
a,b
1,4
</dataframe>




Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd

# Write code here


# Additional code to extract data
# Get current axes
ax = plt.gca()


line = ax.get_lines()[0]
x_data = line.get_xdata().tolist()
y_data = line.get_ydata().tolist()


plot_data = {{'x': x_data, 'y': y_data}}
# Define the path where the data will be saved
file_path = 'D:/AutoDash/innov8/file.json'

# Save the data to a file
with open(file_path, 'w') as file:
    json.dump(plot_data, file)

print(f"Data saved to {{file_path}}")

# Declare result var: 
{output_type_template}

```





Variable `dfs: list[pd.DataFrame]` is already declared.

At the end, declare "result" variable as a dictionary of type and value.


Generate python code and return full updated code:"""  # noqa E501
        actual_prompt_content = prompt.to_string()
        if sys.platform.startswith("win"):
            actual_prompt_content = actual_prompt_content.replace("\r\n", "\n")
        assert actual_prompt_content == expected_prompt_content

    @pytest.mark.parametrize(
        "output_type,output_type_template",
        [
            *[
                (
                    None,
                    """type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "number",
                    """type (must be "number"), value must int. Example: { "type": "number", "value": 125 }""",
                ),
                (
                    "dataframe",
                    """type (must be "dataframe"), value must be pd.DataFrame or pd.Series. Example: { "type": "dataframe", "value": pd.DataFrame({...}) }""",
                ),
                (
                    "plot",
                    """type (must be "plot"), value must be string. Example: { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "string",
                    """type (must be "string"), value must be string. Example: { "type": "string", "value": f"The highest salary is {highest_salary}." }""",
                ),
            ]
        ],
    )
    @patch("pandasai.vectorstores.bamboo_vectorstore.BambooVectorStore")
    def test_str_with_train_qa(self, chromadb_mock, output_type, output_type_template):
        """Test casting of prompt to string and interpolation of context.

        Args:
            output_type (str): output type
            output_type_template (str): output type template

        Returns:
            None
        """

        os.environ["PANDASAI_API_URL"] = "SERVER_URL"
        os.environ["PANDASAI_API_KEY"] = "API_KEY"

        chromadb_instance = chromadb_mock.return_value
        chromadb_instance.get_relevant_qa_documents.return_value = [["query1"]]
        llm = FakeLLM()
        agent = Agent(
            PandasConnector({"original_df": pd.DataFrame({"a": [1], "b": [4]})}),
            config={"llm": llm, "dataframe_serializer": DataframeSerializerType.CSV},
        )
        agent.train(["query1"], ["code1"])
        prompt = GeneratePythonCodePrompt(
            context=agent.context,
            output_type=output_type,
        )

        expected_prompt_content = f"""<dataframe>
dfs[0]:1x2
a,b
1,4
</dataframe>




Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd

# Write code here


# Additional code to extract data
# Get current axes
ax = plt.gca()


line = ax.get_lines()[0]
x_data = line.get_xdata().tolist()
y_data = line.get_ydata().tolist()


plot_data = {{'x': x_data, 'y': y_data}}
# Define the path where the data will be saved
file_path = 'D:/AutoDash/innov8/file.json'

# Save the data to a file
with open(file_path, 'w') as file:
    json.dump(plot_data, file)

print(f"Data saved to {{file_path}}")

# Declare result var: 
{output_type_template}

```


You can utilize these examples as a reference for generating code.

['query1']





Variable `dfs: list[pd.DataFrame]` is already declared.

At the end, declare "result" variable as a dictionary of type and value.


Generate python code and return full updated code:"""  # noqa E501
        actual_prompt_content = prompt.to_string()
        if sys.platform.startswith("win"):
            actual_prompt_content = actual_prompt_content.replace("\r\n", "\n")

        assert actual_prompt_content == expected_prompt_content

    @pytest.mark.parametrize(
        "output_type,output_type_template",
        [
            *[
                (
                    None,
                    """type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "number",
                    """type (must be "number"), value must int. Example: { "type": "number", "value": 125 }""",
                ),
                (
                    "dataframe",
                    """type (must be "dataframe"), value must be pd.DataFrame or pd.Series. Example: { "type": "dataframe", "value": pd.DataFrame({...}) }""",
                ),
                (
                    "plot",
                    """type (must be "plot"), value must be string. Example: { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "string",
                    """type (must be "string"), value must be string. Example: { "type": "string", "value": f"The highest salary is {highest_salary}." }""",
                ),
            ]
        ],
    )
    @patch("pandasai.vectorstores.bamboo_vectorstore.BambooVectorStore")
    def test_str_with_train_docs(
        self, chromadb_mock, output_type, output_type_template
    ):
        """Test casting of prompt to string and interpolation of context.

        Args:
            output_type (str): output type
            output_type_template (str): output type template

        Returns:
            None
        """

        chromadb_instance = chromadb_mock.return_value
        chromadb_instance.get_relevant_docs_documents.return_value = [["query1"]]
        llm = FakeLLM()
        agent = Agent(
            PandasConnector({"original_df": pd.DataFrame({"a": [1], "b": [4]})}),
            config={"llm": llm, "dataframe_serializer": DataframeSerializerType.CSV},
        )
        agent.train(docs=["document1"])
        prompt = GeneratePythonCodePrompt(
            context=agent.context,
            output_type=output_type,
        )

        expected_prompt_content = f"""<dataframe>
dfs[0]:1x2
a,b
1,4
</dataframe>




Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd

# Write code here


# Additional code to extract data
# Get current axes
ax = plt.gca()


line = ax.get_lines()[0]
x_data = line.get_xdata().tolist()
y_data = line.get_ydata().tolist()


plot_data = {{'x': x_data, 'y': y_data}}
# Define the path where the data will be saved
file_path = 'D:/AutoDash/innov8/file.json'

# Save the data to a file
with open(file_path, 'w') as file:
    json.dump(plot_data, file)

print(f"Data saved to {{file_path}}")

# Declare result var: 
{output_type_template}

```





Here are additional documents for reference. Feel free to use them to answer.
['query1']



Variable `dfs: list[pd.DataFrame]` is already declared.

At the end, declare "result" variable as a dictionary of type and value.


Generate python code and return full updated code:"""  # noqa E501
        actual_prompt_content = prompt.to_string()
        if sys.platform.startswith("win"):
            actual_prompt_content = actual_prompt_content.replace("\r\n", "\n")

        assert actual_prompt_content == expected_prompt_content

    @pytest.mark.parametrize(
        "output_type,output_type_template",
        [
            *[
                (
                    None,
                    """type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "number",
                    """type (must be "number"), value must int. Example: { "type": "number", "value": 125 }""",
                ),
                (
                    "dataframe",
                    """type (must be "dataframe"), value must be pd.DataFrame or pd.Series. Example: { "type": "dataframe", "value": pd.DataFrame({...}) }""",
                ),
                (
                    "plot",
                    """type (must be "plot"), value must be string. Example: { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "string",
                    """type (must be "string"), value must be string. Example: { "type": "string", "value": f"The highest salary is {highest_salary}." }""",
                ),
            ]
        ],
    )
    @patch("pandasai.vectorstores.bamboo_vectorstore.BambooVectorStore")
    def test_str_with_train_docs_and_qa(
        self, chromadb_mock, output_type, output_type_template
    ):
        """Test casting of prompt to string and interpolation of context.

        Args:
            output_type (str): output type
            output_type_template (str): output type template

        Returns:
            None
        """

        os.environ["PANDASAI_API_URL"] = "SERVER_URL"
        os.environ["PANDASAI_API_KEY"] = "API_KEY"

        chromadb_instance = chromadb_mock.return_value
        chromadb_instance.get_relevant_docs_documents.return_value = [["documents1"]]
        chromadb_instance.get_relevant_qa_documents.return_value = [["query1"]]
        llm = FakeLLM()
        agent = Agent(
            PandasConnector({"original_df": pd.DataFrame({"a": [1], "b": [4]})}),
            config={"llm": llm},
        )
        agent.train(queries=["query1"], codes=["code1"], docs=["document1"])
        prompt = GeneratePythonCodePrompt(
            context=agent.context,
            output_type=output_type,
        )

        expected_prompt_content = f"""<dataframe>
dfs[0]:1x2
a,b
1,4
</dataframe>




Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd

# Write code here


# Additional code to extract data
# Get current axes
ax = plt.gca()


line = ax.get_lines()[0]
x_data = line.get_xdata().tolist()
y_data = line.get_ydata().tolist()


plot_data = {{'x': x_data, 'y': y_data}}
# Define the path where the data will be saved
file_path = 'D:/AutoDash/innov8/file.json'

# Save the data to a file
with open(file_path, 'w') as file:
    json.dump(plot_data, file)

print(f"Data saved to {{file_path}}")

# Declare result var: 
{output_type_template}

```


You can utilize these examples as a reference for generating code.

['query1']

Here are additional documents for reference. Feel free to use them to answer.
['documents1']



Variable `dfs: list[pd.DataFrame]` is already declared.

At the end, declare "result" variable as a dictionary of type and value.


Generate python code and return full updated code:"""  # noqa E501
        actual_prompt_content = prompt.to_string()
        if sys.platform.startswith("win"):
            actual_prompt_content = actual_prompt_content.replace("\r\n", "\n")

        assert actual_prompt_content == expected_prompt_content

    @patch("pandasai.vectorstores.bamboo_vectorstore.BambooVectorStore")
    def test_str_geenerate_code_prompt_to_json(self, chromadb_mock):
        """Test casting of prompt to string and interpolation of context.

        Args:
            output_type (str): output type
            output_type_template (str): output type template

        Returns:
            None
        """

        chromadb_instance = chromadb_mock.return_value
        chromadb_instance.get_relevant_docs_documents.return_value = [["documents1"]]
        chromadb_instance.get_relevant_qa_documents.return_value = [["query1"]]
        llm = FakeLLM()
        agent = Agent(
            PandasConnector({"original_df": pd.DataFrame({"a": [1], "b": [4]})}),
            config={"llm": llm},
        )
        agent.train(queries=["query1"], codes=["code1"], docs=["document1"])
        prompt = GeneratePythonCodePrompt(
            context=agent.context, viz_lib="", output_type=None
        )
        prompt_json = prompt.to_json()
        if sys.platform.startswith("win"):
            prompt_json["prompt"] = prompt_json["prompt"].replace("\r\n", "\n")

        assert prompt_json == {
            "datasets": [
                {"name": None, "description": None, "head": [{"a": 1, "b": 4}]}
            ],
            "conversation": [],
            "system_prompt": None,
            "prompt": '<dataframe>\ndfs[0]:1x2\na,b\n1,4\n</dataframe>\n\n\n\n\nUpdate this initial code:\n```python\n# TODO: import the required dependencies\nimport pandas as pd\n\n# Write code here\n\r# Additional code to extract data\r# Get current axes\rax = plt.gca()\r\r\rline = ax.get_lines()[0]\rx_data = line.get_xdata().tolist()\ry_data = line.get_ydata().tolist()\r\r\rplot_data = {{"x": x_data, "y": y_data}}\r# Define the path where the data will be saved\rfile_path = "D:/AutoDash/innov8/file.json"\r\r# Save the data to a file\rwith open(file_path, "w") as file:\r    json.dump(plot_data, file)\r\rprint(f\"Data saved to {{file_path}}\")\n# Declare result var: \ntype (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }\n\n```\n\n\nYou can utilize these examples as a reference for generating code.\n\n[\'query1\']\n\nHere are additional documents for reference. Feel free to use them to answer.\n[\'documents1\']\n\n\n\nVariable `dfs: list[pd.DataFrame]` is already declared.\n\nAt the end, declare "result" variable as a dictionary of type and value.\n\n\nGenerate python code and return full updated code:',
            "config": {"direct_sql": False, "viz_lib": "", "output_type": None},
        }

    @pytest.mark.parametrize(
        "output_type,output_type_template",
        [
            *[
                (
                    None,
                    """type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "number",
                    """type (must be "number"), value must int. Example: { "type": "number", "value": 125 }""",
                ),
                (
                    "dataframe",
                    """type (must be "dataframe"), value must be pd.DataFrame or pd.Series. Example: { "type": "dataframe", "value": pd.DataFrame({...}) }""",
                ),
                (
                    "plot",
                    """type (must be "plot"), value must be string. Example: { "type": "plot", "value": "temp_chart.png" }""",
                ),
                (
                    "string",
                    """type (must be "string"), value must be string. Example: { "type": "string", "value": f"The highest salary is {highest_salary}." }""",
                ),
            ]
        ],
    )
    @patch("pandasai.vectorstores.bamboo_vectorstore.BambooVectorStore")
    def test_str_relations(self, chromadb_mock, output_type, output_type_template):
        """Test casting of prompt to string and interpolation of context.

        Args:
            output_type (str): output type
            output_type_template (str): output type template

        Returns:
            None
        """

        os.environ["PANDASAI_API_URL"] = "SERVER_URL"
        os.environ["PANDASAI_API_KEY"] = "API_KEY"

        chromadb_instance = chromadb_mock.return_value
        chromadb_instance.get_relevant_qa_documents.return_value = [["query1"]]
        llm = FakeLLM()
        agent = Agent(
            PandasConnector(
                {"original_df": pd.DataFrame({"a": [1], "b": [4]})},
                connector_relations=[PrimaryKey("a")],
            ),
            config={"llm": llm, "dataframe_serializer": DataframeSerializerType.CSV},
        )
        agent.train(["query1"], ["code1"])
        prompt = GeneratePythonCodePrompt(
            context=agent.context,
            output_type=output_type,
        )

        expected_prompt_content = f"""dfs[0]:
  name: null
  description: null
  type: pd.DataFrame
  rows: 1
  columns: 2
  schema:
    fields:
    - name: a
      type: int64
      samples:
      - 1
      constraints: PRIMARY KEY (a)
    - name: b
      type: int64
      samples:
      - 4




Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd

# Write code here


# Additional code to extract data
# Get current axes
ax = plt.gca()


line = ax.get_lines()[0]
x_data = line.get_xdata().tolist()
y_data = line.get_ydata().tolist()


plot_data = {{'x': x_data, 'y': y_data}}
# Define the path where the data will be saved
file_path = 'D:/AutoDash/innov8/file.json'

# Save the data to a file
with open(file_path, 'w') as file:
    json.dump(plot_data, file)

print(f"Data saved to {{file_path}}")

# Declare result var: 
{output_type_template}

```


You can utilize these examples as a reference for generating code.

['query1']





Variable `dfs: list[pd.DataFrame]` is already declared.

At the end, declare "result" variable as a dictionary of type and value.


Generate python code and return full updated code:"""  # noqa E501
        actual_prompt_content = prompt.to_string()
        if sys.platform.startswith("win"):
            actual_prompt_content = actual_prompt_content.replace("\r\n", "\n")

        print(actual_prompt_content)

        assert actual_prompt_content == expected_prompt_content
