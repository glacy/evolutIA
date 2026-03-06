## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-24 - Pre-indexing for O(1) Lookups in Nested Loops
**Learning:** Found an $O(N \times M)$ nested loop in `MaterialExtractor.get_all_exercises` that was matching exercises to solutions by iterating over all solutions for every exercise.
**Action:** Replaced the inner loop with a dictionary lookup by pre-indexing the solutions by `exercise_label`. This turns an $O(N \times M)$ operation into an $O(N + M)$ operation, significantly improving performance when parsing large documents with many exercises.
