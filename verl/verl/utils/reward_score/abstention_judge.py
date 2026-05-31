"""
Abstention Judge - Reward function for abstention RL training.

Handles three data sources:
1. math_dapo: answerable math questions, evaluated by math_verify
2. abstention_insufficient: questions with insufficient info, expected answer "I don't know"
3. abstention_conflict: questions with false premise / conflicting info, expected answer "False premise"

For abstention types, uses string matching on the content inside \\boxed{}.
"""

import re


def compute_score(solution_str, ground_truth) -> float:
    """
    Compute reward score for abstention tasks.

    For abstention (ground_truth in ["I don't know", "False premise"]):
      - Extract content from the last \\boxed{} in the solution.
      - Normalize and compare with ground_truth via string matching.
      - Correct abstention answer: 1.0 + format bonus
      - Wrong answer: 0.0 + format bonus

    Format bonus: +0.5 if the solution contains a valid \\boxed{}.

    Args:
        solution_str: The model's generated solution string.
        ground_truth: The expected answer string.

    Returns:
        float: The computed score.
    """
    retval = 0.0
    try:
        boxed_content = last_boxed_only_string(solution_str)

        if boxed_content is not None:
            answer = normalize_answer(remove_boxed(boxed_content))
            gt_norm = normalize_answer(ground_truth)

            if answer == gt_norm:
                retval += 1.0

        # Format reward: has a valid \boxed{}
        if check_format(solution_str):
            retval += 0.2
    except Exception as e:
        print(f"[abstention_judge] Error: {e}")

    return retval


def normalize_answer(ans: str) -> str:
    """
    Normalize answer string for comparison.
    - Strip whitespace
    - Remove \\text{} wrappers
    - Lowercase
    - Remove extra spaces
    """
    ans = ans.strip()
    # Remove LaTeX \text{} wrapping
    ans = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", ans)
    # Remove spaces and lowercase
    ans = ans.replace(" ", "").lower()
    return ans


def check_format(solution_str, max_chars_after_boxed: int = 1000) -> bool:
    """
    Check if the solution contains a valid \\boxed{} and it's near the end.
    """
    if "\\boxed{" not in solution_str and "\\boxed " not in solution_str:
        return False

    if max_chars_after_boxed is not None:
        idx = solution_str.rfind("\\boxed")
        if idx < 0:
            idx = solution_str.rfind("\\fbox")
        if idx >= 0:
            after = solution_str[idx:]
            if len(after) > max_chars_after_boxed:
                return False

    return True


def last_boxed_only_string(string):
    """Extract the last \\boxed{...} from the string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx: right_brace_idx + 1]
    return retval


def remove_boxed(s):
    """Remove the \\boxed{} wrapper and return the content inside."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left): -1]
