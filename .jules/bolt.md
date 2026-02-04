## 2024-05-23 - Regex Compilation Optimization
**Learning:** When combining regex patterns with `(?:...)` that contain global flags like `(?i)` at the start, `re.compile` will fail with "global flags not at the start of the expression" even if the flags are redundant.
**Action:** Strip `(?i)` from individual patterns before combining them if using `re.IGNORECASE` flag on the combined regex.
