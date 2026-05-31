def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        solution = solution_str.split("Label:")[-1]
        if solution is not None:
            answer = normalize_answer(solution)
            gt_norm = normalize_answer(ground_truth)
            if answer == gt_norm:
                retval += 1.0
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
