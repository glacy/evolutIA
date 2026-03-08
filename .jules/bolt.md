## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-24 - O(N*M) Lookup Optimization
**Learning:** Found nested loops performing lookups to match exercises with their corresponding solutions in `MaterialExtractor.get_all_exercises` and `RAGIndexer.index_materials`. The time complexity was $O(N \times M)$ where N is exercises and M is solutions. In Python, this becomes a severe bottleneck as dataset sizes grow.
**Action:** Replaced nested loops with a pre-computed dictionary mapping (hash map) using `solutions_by_label = {sol['exercise_label']: sol for sol in ...}`. This reduces lookup time from $O(M)$ to $O(1)$ per exercise, bringing the total complexity to $O(N + M)$.
