## 2025-05-15 - Regex Alternation Performance
**Learning:** Replacing multiple `re.search()` calls for simple literals with a single `re.compile(r'literal1|literal2|...')` regex was ~50% SLOWER in Python.
**Insight:** Python's `re` module likely uses optimized string search algorithms (like Boyer-Moore) for simple literal patterns, which are faster than the state machine overhead of a large alternation regex.
**Action:** Prefer multiple simple `re.search()` calls over complex alternations when patterns are mostly literals. Only use combined regex when tokenization/parsing requires strictly ordered matching or when patterns share complex prefixes.

## 2025-05-20 - Pre-compiling Regex in Loops
**Learning:** `re.findall(pattern, string)` recompiles (or retrieves from cache) the pattern on every call. In high-frequency functions called inside loops (like complexity estimation), this overhead adds up.
**Action:** Always pre-compile regexes (`re.compile`) into module-level or class-level constants if they are used repeatedly, especially in tight loops or recursive functions.

## 2025-02-12 - [Fix unreachable cache update in `MaterialExtractor`]
**Learning:** Found a critical anti-pattern where a method `extract_from_file` assigned its extracted result directly to a `return { ... }` statement *before* the caching block `# Guardar en caché`. This bypassed the entire caching mechanism completely since the function exited prematurely, breaking the `_file_cache` optimization.
**Action:** When implementing or optimizing caching logic, always ensure the return statement is placed *after* the cache has been updated and that the result object is correctly captured in a variable (e.g., `result = { ... }`) to be passed to both the cache and the caller.
