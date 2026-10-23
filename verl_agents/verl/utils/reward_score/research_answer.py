"""Strict final-answer extraction for the research numeric reward, not evaluation."""
import math
import re


def final_numeric_answer(response):
    # Ignore tool observations and earlier assistant turns. Never scan reasoning
    # for a number when the final answer is absent or malformed.
    turns = re.findall(r'<\|im_start\|>assistant\s*\n(.*?)(?=<\|im_start\|>|\Z)', response, re.S)
    text = turns[-1] if turns else response
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    if not answers:
        return None
    text = answers[-1].strip()
    boxes = re.findall(r'\\boxed\s*\{([^{}]*)\}', text, re.S)
    if boxes:
        text = boxes[-1].strip()
    # Permit one explicit physical unit; do not silently accept extra numbers.
    match = re.fullmatch(r'\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:meters?|metres?|m|cm|mm|seconds?|s|degrees?|°)?\s*', text, re.I)
    if not match:
        return None
    value = float(match.group(1))
    # Numeric labels are in benchmark units. Unit conversion is deliberately
    # not guessed; cm/mm require explicit normalization in a future experiment.
    if re.search(r'\b(?:cm|mm)\s*$', text, re.I):
        return None
    return value if math.isfinite(value) else None
