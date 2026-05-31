def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = normalize_answer(remove_boxed(string_in_last_boxed))
            gt_norm = normalize_answer(ground_truth)
            if answer == gt_norm:
                retval += 1.0
        # format-reward check
        if check_format(solution_str):
            retval += 0.5  # format ok but answer mismatch: partial 0.5
    except Exception as e:
        print(e)
    return retval

def normalize_answer(ans: str) -> str:
    """
    normalize 'Path1', '\text{path1}', '\text{Path1}' to 'path1'
    """
    import re
    ans = ans.strip()
    # strip LaTeX \text{} wrappers
    ans = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", ans)
    # remove spaces and lowercase
    ans = ans.replace(" ", "").lower()
    return ans

def check_format(solution_str, allow_multiple_boxed: bool = True, max_chars_after_boxed: int = 1000) -> bool:
    """
    check whether the answer follows the required format
    
    Args:
        solution_str: string to check
        allow_multiple_boxed: whether multiple \boxed{} are allowed (default True for discrimination tasks)
        max_chars_after_boxed: max characters allowed after the last \boxed{} (default 1000)
    
    Returns:
        whether the format is valid
    """
    # 1. must contain a \boxed{} match
    if "\\boxed{" not in solution_str and "\\boxed " not in solution_str:
        return False
    
    # 2. characters after \boxed{} must not exceed the limit (if set)
    if max_chars_after_boxed is not None:
        idx = solution_str.rfind("\\boxed")
        if idx < 0:
            idx = solution_str.rfind("\\fbox")
        if idx >= 0:
            after = solution_str[idx:]
            if len(after) > max_chars_after_boxed:
                return False
    
    # 3. \boxed{} must be unique (unless multiple allowed)
    if not allow_multiple_boxed:
        count = solution_str.count("\\boxed{") + solution_str.count("\\boxed ")
        if count != 1:
            return False
    
    return True

def last_boxed_only_string(string):
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

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]