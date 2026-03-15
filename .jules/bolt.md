## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-25 - O(N^2) Nested Loops for Relations
**Learning:** Found multiple instances where relationships between two sets of data within the same file (e.g., matching exercises to solutions by a shared label) were implemented using nested loops, leading to O(N*M) time complexity.
**Action:** Replace nested loops that search for matching keys with a single pass to build a lookup dictionary O(N), followed by an O(1) lookup during the second iteration, reducing overall complexity to O(N+M). This provides a significant speedup for large documents.
