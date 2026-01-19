from __future__ import annotations


def get_system_prompt(dataset: str) -> str:
    """Return the system prompt used for premise selection.

    This is shared between API evaluation and local prompting baselines.

    Args:
        dataset: Either "base" or "meta".

    Returns:
        The system prompt string.
    """
    if dataset == "meta":
        return (
            "You are tasked with logical premise selection. Given:\n"
            "1. A knowledge base consisting of premises.\n"
            "2. Example hypotheses along with their correct minimal premise sets, preceded by the token <STUDY>.\n"
            "3. A query hypothesis to solve, preceded by the token <QUERY>.\n\n"
            "Your task is to identify the unique minimal set of premises from the knowledge base that logically proves the query hypothesis. "
            "Since the knowledge base is non-redundant, every valid hypothesis has exactly one minimal set of premises that proves it.\n\n"
            "Examine the provided examples carefully to understand how to select the correct minimal set of premises. "
            "The examples demonstrate correct premise selections for various hypotheses.\n\n"
            "Provide your answer in exactly this format:\n"
            "### Answer: premise1, premise2, ..., premiseN"
        )

    if dataset == "base":
        return (
            "You are tasked with logical premise selection. Given:\n"
            "1. A knowledge base consisting of premises.\n"
            "2. A query hypothesis to solve, preceded by the token <QUERY>.\n\n"
            "Your task is to identify the unique minimal set of premises from the knowledge base that logically proves the query hypothesis. "
            "Since the knowledge base is non-redundant, every valid hypothesis has exactly one minimal set of premises that proves it.\n\n"
            "Provide your answer in exactly this format:\n"
            "### Answer: premise1, premise2, ..., premiseN"
        )

    raise ValueError("dataset must be 'base' or 'meta'")
