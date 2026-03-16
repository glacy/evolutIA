## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-02-28 - Regex extraction optimization via string concatenation
**Learning:** In Python, calling `re.finditer` inside a `for` loop over a list of strings adds significant Python-level overhead compared to running the regex engine on a single large string. Furthermore, directly replacing `finditer` with `findall` to eliminate loops breaks group semantics (returns a tuple rather than a match object) and requires further logic that negates the speedup.
**Action:** When extracting grouped data from an iterable of strings (like `extract_variables` iterating over math expressions), optimize by concatenating the strings with `" ".join(strings)` and running a single `finditer` pass on the combined string. This reduces execution time significantly while preserving group index semantics safely.

## 2025-02-28 - Accidental full-file rewriting via LF/CRLF conversion
**Learning:** Using tools that completely alter file line endings (e.g., `unix2dos` or `dos2unix`) or replacing large chunks of code with differing line endings corrupts Git history, causing `git diff` to show the entire file as modified rather than just the targeted lines.
**Action:** Always verify line ending formats before applying broad edits and restrict modifications purely to the specific logical blocks needed to preserve `git blame` and minimize diff noise.

## 2025-02-28 - First vs. Last element resolution in O(N) lookup conversions
**Learning:** When refactoring a nested loop $O(N \times M)$ search that uses a `break` statement into an $O(N)$ lookup using a dictionary comprehension `d = {item['id']: item for item in items}`, the dictionary comprehension implicitly stores the *last* matching element for any duplicate keys, whereas the original loop stopped at the *first* match.
**Action:** When creating dictionaries to replace search loops, ensure that duplicate keys are either impossible (e.g., guaranteed unique IDs) or explicitly handled by reversing the input list or using a standard `for` loop that checks `if key not in d:` to match original semantics exactly.
