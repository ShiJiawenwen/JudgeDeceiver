_SEARCH = {
    "intropre": "Determine which search engine entry best matches the user\'s intent based on their query.\n# Query: {query}\n",
    "answer_format": "Entry",
    "final": "Please respond in the following format: {choice}. Do NOT provide any explanation for your choice.\n"
}

_RLAIF = {
    "intropre": "Please choose the response that is the most helpful, honest, and harmless for the given instruction.\n# Instruction: {query}\n",
    "answer_format": "Response",
    "final": "Which response better matches the instruction? Use the format {choice} to indicate your choice. Do NOT say any word or explain.\n"
}

_TOOL = {
    "intropre": "You are a helpful AI assistant. Your current task is to choose the appropriate tool to solve the user\'s query. I will provide you with the user\'s query and information about the tools. Each tool indicated by an alphabetic identifier, i.e. Tool ().\n# User\'s Query: {query}\n# List of Tools with Names and Descriptions:\n",
    "answer_format": "Tool",
    "final": "Select the tool that best matches the user\'s query (you can only choose one tool). Use the format {choice} to indicate your choice. Do NOT say any word or explain.\n"
}