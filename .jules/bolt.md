## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-02-18 - [Regex Optimization: finditer vs findall vs Combining]
**Learning:**  is significantly faster (~20-25%) than  for simple extraction tasks in Python because the iteration happens in C. However, combining multiple independent regexes into a single complex regex (OR-ed) and using  to scan once can be SLOWER than multiple  passes if the original  calls exit early (short-circuit).
**Action:** Prefer  for extraction loops. Avoid combining simple existence checks () into a single complex  loop unless you need counts of all occurrences.

## 2025-02-18 - [Regex Optimization: finditer vs findall vs Combining]
**Learning:** `re.findall` is significantly faster (~20-25%) than `re.finditer` for simple extraction tasks in Python because the iteration happens in C. However, combining multiple independent regexes into a single complex regex (OR-ed) and using `finditer` to scan once can be SLOWER than multiple `search` passes if the original `search` calls exit early (short-circuit).
**Action:** Prefer `findall` for extraction loops. Avoid combining simple existence checks (`search`) into a single complex `finditer` loop unless you need counts of all occurrences.
