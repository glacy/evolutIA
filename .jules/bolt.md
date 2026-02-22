## 2024-05-22 - Regex Optimization Performance
**Learning:** Python's `re.findall` is significantly faster (20-50%) than `re.finditer` loops when extracting simple patterns, because it builds the list in C. However, for complex regexes with many groups where you need to know *which* group matched, `match.lastindex` is much faster than checking `match.group('name')` or iterating over groups in Python.
**Action:** Prefer `findall` for extraction tasks. If `finditer` is necessary (e.g. for match positions), use `match.lastindex` to access the matched group instead of named lookups.
