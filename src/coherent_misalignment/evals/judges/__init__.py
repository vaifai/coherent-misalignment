"""Judge backends for the harm evaluation rubric.

Each backend exports a `make_*_judge(...)` factory returning a `JudgeFn`,
i.e. a `(prompt: str, response: str) -> Optional[int]` callable returning
a 1-5 score or None on parse-fail.

The factory is the same shape as `harm_eval.make_openai_judge` so that
new backends can be slotted into the same orchestration code without
changes upstream.
"""
