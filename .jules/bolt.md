## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-22 - Python Loop Optimization: Hoisting attributes and len()
**Learning:** In tight loops with a large number of iterations (like generating many exercises), repeatedly calling `len()` and looking up object attributes (e.g. `args.tema`, `args.tags`) adds measurable overhead.
**Action:** Always hoist length calculations and static attribute lookups outside the loop for high-frequency iteration blocks to avoid redundant function calls and dictionary lookups.

## 2025-05-22 - re.findall vs re.finditer Overhead
**Learning:** `re.findall` avoids the `re.Match` object instantiation overhead compared to `re.finditer` when only string values are needed. In our benchmarks, using `findall` over `finditer` for regex extraction tasks (like Markdown parsing) yields a ~20-30% speedup.
**Action:** Always prefer `re.findall` over `re.finditer` when the match groups' string values are the only required outputs, bypassing the overhead of creating match objects.
