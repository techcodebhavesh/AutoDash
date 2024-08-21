from .tempbase import BasePrompt


class GenerateScatterPlot(BasePrompt):
    """Prompt to generate Python code from a dataframe."""
    print("Initializing Scatter Plot Generation")

    template_path = "generateScatterPlot.tmpl"

    def to_json(self):
        print("Converting to JSON")
        context = self.props["context"]
        print("Context retrieved")
        code = self.props["code"]
        error = self.props["error"]
        memory = context.memory
        conversations = memory.to_json()

        system_prompt = memory.get_system_prompt()

        # prepare datasets
        datasets = [dataset.to_json() for dataset in context.dfs]

        print("Context:", context)
        print("Code:", code)
        print("Error:", error)
        print("Memory:", memory)
        print("Conversations:", conversations)
        print("System Prompt:", system_prompt)

        return {
            "datasets": datasets,
            "conversation": conversations,
            "system_prompt": system_prompt,
            "error": {
                "code": code,
                "error_trace": str(error),
                "exception_type": "Exception",
            },
            "config": {"direct_sql": context.config.direct_sql},
        }
