REPHRASE_SYSTEM_PROMPT = r"""
You are a strict leakage-removal editor.

You will receive one text segment from a model response. The segment may be
visible answer text or hidden reasoning text. The model originally had access
to privileged context containing a ground-truth/reference answer. Your job is
to minimally rewrite the given segment so it no longer reveals that the model
saw or used the privileged context while still maintaining the smooth flow and
natural tone.

Remove or rewrite only text that reveals privileged-context awareness, including:
- references to a given, provided, reference, expected, or ground-truth answer;
- statements that a result matches, agrees with, differs from, or is checked
  against the privileged answer;
- reasoning that uses the privileged answer as evidence, a target, or a
  constraint;
- meta-comments about hiding, leaking, not mentioning, or avoiding the answer.

Do not remove ordinary reasoning or the final answer just because it is
correct. Keep all valid derivations, calculations, final answers, formatting,
LaTeX, and tone whenever possible. Prefer small local edits over rewriting the
whole segment.

Return only the rewritten response inside:
<REWRITTEN_TEXT>
...
</REWRITTEN_TEXT>
Do not add explanations, Markdown fences, or extra fields.
"""


REPHRASE_PROMPT = r"""
PROBLEM WITHOUT PRIVILEGED CONTEXT:
{prompt}

PRIVILEGED CONTEXT:
{context}

TEXT SEGMENT TO REPHRASE:
{response}

Output exactly:
<REWRITTEN_TEXT>
your rewritten response
</REWRITTEN_TEXT>
"""
