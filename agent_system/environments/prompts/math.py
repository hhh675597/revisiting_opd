MATH_TEMPLATE = """Math problem: {task_description}

Please carefully reason through the math problem step by step and derive the correct answer. You should give the final answer within \\boxed{{}}.
"""

# with <think> and </think> version
# MATH_TEMPLATE = """Math problem: {task_description}

# Please carefully reason through the math problem step by step and derive the correct answer. You must conduct reasoning inside <think> and </think> and give the final answer within \\boxed{{}}.
# """