## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-03-03 - Avoiding `re.Match` Overhead with `findall`
**Learning:** When extracting values with regex capture groups using `re.finditer` inside a tight loop, the creation of `re.Match` objects adds significant overhead. Replacing `pattern.finditer(string)` with `pattern.findall(string)` is ~30-40% faster in Python when only the matched string values are needed, because `findall` returns primitive tuples of strings.
**Action:** Use `re.findall` instead of `re.finditer` when you only need to extract the matched group values and do not need match objects (for spans, advanced group dicts, etc.).
