## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2024-05-18 - [Optimization] Joining iterables before applying RegEx speeds up parsing overhead
**Learning:** In string parsing functions, joining small expressions inside a list into a single large string and processing it using `re.finditer` once is much faster than repeatedly invoking `re.finditer` inside a loop over the list elements, avoiding excessive looping and regex execution overhead.
**Action:** When extracting components (like variables or expressions) from a list of mathematical expressions or strings, safely join the list together if token overlap is avoided by regex rules, and parse the concatenated string.
