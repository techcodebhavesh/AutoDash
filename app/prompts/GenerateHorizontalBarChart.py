from .tempbase import BasePrompt


class GenerateHorizontalBar(BasePrompt):
    """Prompt to generate Python code from a dataframe for a horizontal bar chart."""
    
    template_path = "generateHorizontalBarPro.tmpl"

    def to_json(self):
        context = self.props["context"]
        code = self.props["code"]
        error = self.props["error"]
        memory = context.memory
        conversations = memory.to_json()

        system_prompt = memory.get_system_prompt()

        # prepare datasets
        datasets = [dataset.to_json() for dataset in context.dfs]

        return {
            "datasets": datasets,
            "conversation": conversations,
            "system_prompt": system_prompt,
            "error": {
                "code": code,
                "error_trace": str(error),
                "exception_type": "Exception",
            },
            "config": {"direct_sql": context.config.direct_sql, "chart_type": "horizontal_bar"},
        }