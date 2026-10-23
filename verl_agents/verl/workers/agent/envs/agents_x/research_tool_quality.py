"""Reward successful, nonempty, nonduplicate tool feedback, capped per episode."""
import ast
import functools
import hashlib
import re


def reset_quality_state(tool):
    tool._research_codes = set()
    tool._research_feedback = set()
    tool._research_credited = 0


def quality_reward(execute):
    @functools.wraps(execute)
    def wrapped(tool, action_string, **kwargs):
        obs, reward, done, info = execute(tool, action_string, **kwargs)
        import os
        if 'tool' not in os.environ.get('PYVISION_ABLATION_FEATURES', 'numeric,quota,tool').split(','):
            return obs, reward, done, info
        info = dict(info)
        if not hasattr(tool, '_research_codes'):
            reset_quality_state(tool)
        reason = 'not_successful_tool_output'
        bonus = 0.0
        if info.get('status') == 'success' and not info.get('error') and not done:
            code = tool.extract_code(action_string) or ''
            try:
                normalized = ast.dump(ast.parse(code), include_attributes=False)
            except SyntaxError:
                normalized = code.strip()
            code_hash = hashlib.sha256(normalized.encode()).hexdigest()
            images = obs.get('multi_modal_data', {}).get('image', []) if isinstance(obs, dict) else []
            if images:
                payload = b''.join(str((im.mode, im.size)).encode() + im.tobytes() for im in images)
            else:
                text = re.search(r'<interpreter>(.*?)</interpreter>', str(obs), re.S)
                text = text.group(1) if text else ''
                text = re.sub(r'^\s*Text Result:\s*', '', text).strip()
                payload = text.encode() if text not in ('', 'None') else b''
            feedback_hash = hashlib.sha256(payload).hexdigest()
            if not payload:
                reason = 'empty_output'
            elif code_hash in tool._research_codes or feedback_hash in tool._research_feedback:
                reason = 'duplicate_code_or_output'
            elif tool._research_credited >= 2:
                reason = 'episode_bonus_cap'
            else:
                bonus = float(tool.tool_using_cumulative_reward_per_turn)
                tool._research_credited += 1
                reason = 'credited'
            tool._research_codes.add(code_hash)
            tool._research_feedback.add(feedback_hash)
        info.update(research_tool_bonus=bonus, research_tool_bonus_reason=reason)
        return obs, bonus, done, info
    return wrapped
