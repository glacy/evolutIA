## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-21 - Optimizing Nested Loops to Hash Maps
**Learning:** Found an O(N*M) nested loop used to map solutions to exercises in `evolutia/material_extractor.py` and `evolutia/rag/rag_indexer.py`.
**Insight:** A micro-benchmark showed that replacing this nested iteration with an O(1) dictionary lookup (resulting in O(N+M) complexity) yields a ~50x speedup when processing lists of 1000 items.
**Action:** Always look for O(N^2) patterns when associating elements from two lists and replace them with hash map (dict) lookups.
