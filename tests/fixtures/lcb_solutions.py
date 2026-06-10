"""Hand-written, harness-verified reference solutions for the LCB fixture.

Keyed by question_id; each value is a full generation string with the
fenced ```python code block (exercises extract_code). Verified on 2026-06-04.
"""

SOLUTIONS = {
    'abc333_a': '```python\nn = input().strip()\nprint(n * int(n))\n```',
    '2819': '```python\nclass Solution:\n    def removeTrailingZeros(self, num):\n        return num.rstrip("0")\n```',
    'abc387_a': '```python\na, b = map(int, input().split())\nprint((a + b) ** 2)\n```',
    '3747': '```python\nclass Solution:\n    def maxAdjacentDistance(self, nums):\n        n = len(nums)\n        return max(abs(nums[i] - nums[(i + 1) % n]) for i in range(n))\n```',
}

WRONG = "```python\nprint('definitely wrong output')\n```"
