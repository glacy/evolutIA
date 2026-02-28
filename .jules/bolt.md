## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-24 - Group Extraction with Regex
**Learning:** For regex patterns utilizing alternations with named or positional capture groups where the only goal is to extract the truthy group value (no position info needed), `re.findall` is faster (~15-35%) than using `re.finditer` inside a loop checking each `.group()`.
**Insight:** `re.finditer` constructs a full `re.Match` object per match. When using multiple capture groups, iterating over tuples of groups natively from `findall()` is significantly cheaper on the Python side than object instantiations from `finditer`.
**Action:** When extracting group matched strings specifically, use `re.findall` and loop through the returned tuple values rather than creating `Match` objects with `re.finditer`.
