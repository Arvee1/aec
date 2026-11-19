# Sidebar LLM model indicator

Date: 2026-11-19

## Intent
Show users which language model the `aecreformauth.py` app uses, so the active model is transparent.

## Goals and non-goals
- Display the exact Replicate model identifier in the sidebar.
- Keep the displayed identifier and the model sent to Replicate tied to one configuration value.
- Do not change the selected model or generation settings.
- Do not update other apps in the repository.

## Approaches considered
1. Add a second, static model string to the sidebar. This is the smallest edit but can become inaccurate if the API call changes.
2. Add a shared `Config.LLM_MODEL` value, use it in the Replicate call, and display it in the sidebar. This is the selected approach because the sidebar will reflect the actual configured model.

## Design
Add `LLM_MODEL` to the app configuration, replace the model literal in `LLaMAClient.generate_response` with that value, and display a labeled model identifier in the existing sidebar near document management. Keep the display visible regardless of whether the user has entered a name. The active identifier remains `meta/meta-llama-3-70b-instruct`.

## Verification
Use test-first development to verify the shared model identifier/display behavior with the repository's `unittest` setup, then run the full test suite and Python compile check. No third-party dependency is needed, so `requirements.txt` remains unchanged.
