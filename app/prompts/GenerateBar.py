from .tempbase import BasePrompt


class GenerateBar(BasePrompt):
    """Prompt to generate Python code from a dataframe."""
    print("jsizna")

    template_path = "generateBarPro.tmpl"

    def to_json(self):
        
        
        
        print("hwbsbhjs")
        context = self.props["context"]
        print("huu")
        code = self.props["code"]
        error = self.props["error"]
        memory = context.memory
        conversations = memory.to_json()

        system_prompt = memory.get_system_prompt()

        # prepare datasets
        datasets = [dataset.to_json() for dataset in context.dfs]

        print(context)
        print(code)
        print(error)
        print(memory)
        print(conversations)
        print(system_prompt)


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
