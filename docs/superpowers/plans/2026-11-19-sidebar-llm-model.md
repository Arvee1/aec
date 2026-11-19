# Sidebar LLM Model Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the exact LLM model used by `aecreformauth.py` in its sidebar.

**Architecture:** Keep the model identifier in a small, dependency-free settings module so its value and display label can be tested without importing Streamlit or the app's data stack. Expose it as `Config.LLM_MODEL`, and use that same value for both `replicate.stream` and the sidebar label.

**Tech Stack:** Python standard library, Streamlit, Replicate, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-19-sidebar-llm-model-design.md`

## Global Constraints

- Keep the model identifier `meta/meta-llama-3-70b-instruct` unchanged.
- Use one configured model value for the Replicate call and sidebar label.
- Do not change LLM generation settings or other applications.
- Do not add third-party dependencies; leave `requirements.txt` unchanged.
- Work on `feat/sidebar-llm-model`; never push directly to `main`; open a PR to `main` for review.

## Review Focus

- If the configured model value is blank, the sidebar should not silently present a different model; the formatter should display the configured value verbatim.
- The sidebar label should include the exact configured identifier; cover in `test_format_llm_model_label_shows_identifier`.
- `Config.LLM_MODEL` should use the single model identifier consumed by Replicate; cover with a source-level wiring test that checks the config and API call.
- The sidebar should show the model even before a user enters a name; cover by checking `render_sidebar` references the configured value outside the authenticated-only block.
- The existing model ID must remain unchanged; cover in `test_configured_model_id`.

---

### Task 1: Centralize and display the model identifier

**Files:**
- Create: `llm_settings.py` — dependency-free model ID and label formatter.
- Modify: `aecreformauth.py` — add `Config.LLM_MODEL`, pass it to Replicate, and display it in the sidebar.
- Create: `tests/test_llm_settings.py` — unit tests for model ID, label, and application wiring.

**Interfaces:**
- `llm_settings.LLM_MODEL_ID: str` is `meta/meta-llama-3-70b-instruct`.
- `llm_settings.format_llm_model_label(model_id: str) -> str` returns `LLM model: <model_id>` with the provided value unchanged.
- `Config.LLM_MODEL` aliases `LLM_MODEL_ID`; the Replicate stream and sidebar both consume `self.config.LLM_MODEL`.

- [ ] **Step 1: Write the failing tests**

Add `test_configured_model_id`, `test_format_llm_model_label_shows_identifier`, `test_format_llm_model_label_preserves_empty_value`, and `test_app_uses_configured_model_for_api_and_sidebar`. The last test parses `aecreformauth.py` with `ast` so the tests do not import unavailable Streamlit/ChromaDB dependencies; assert that `Config.LLM_MODEL` references `LLM_MODEL_ID`, the Replicate model argument references `self.config.LLM_MODEL`, and the sidebar displays that same configured value before its authentication-only section.

- [ ] **Step 2: Run the new tests and verify the expected failure**

Run: `python -m unittest discover -s tests -p 'test_llm_settings.py' -v`
Expected: FAIL because `llm_settings` and the shared application setting do not yet exist.

- [ ] **Step 3: Implement the minimal settings and UI changes**

Create `llm_settings.py` with the specified constant and formatter. Add `Config.LLM_MODEL = LLM_MODEL_ID`, change only the model argument to `replicate.stream` to use `self.config.LLM_MODEL`, and add a sidebar caption using `format_llm_model_label(self.config.LLM_MODEL)` near Document Management.

- [ ] **Step 4: Run focused and full verification**

Run: `python -m unittest discover -s tests -p 'test_llm_settings.py' -v`
Expected: PASS.

Run: `python -m unittest discover -s tests -v`
Expected: all tests pass, including existing name-validation tests.

Run: `python -m compileall -q aecreformauth.py llm_settings.py tests`
Expected: exit code 0.

- [ ] **Step 5: Review, commit, push, and open a PR**

Commit on `feat/sidebar-llm-model`, push that branch, and open a PR targeting `main`; do not merge it without Ryan's request.
