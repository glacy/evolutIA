## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-05-25 - Regex Match Group Lookup Optimization
**Learning:** Checking `match.group(name)` for multiple alternatives sequentially involves dictionary lookups and function calls for each non-matching group. Using `match.lastgroup` to directly access the single matching group avoids N-1 failed lookups.
**Insight:** For regexes with many mutually exclusive named groups (like `(?P<A>...)|(?P<B>...)|...`), `match.lastgroup` provides O(1) access to the successful match, whereas sequential checks are O(N).
**Action:** Use `match.lastgroup` when processing matches from a regex composed of top-level named alternatives.
