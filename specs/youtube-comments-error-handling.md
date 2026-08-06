---
title: Handle errors in get_youtube_comments instead of crashing
status: completed
created: 2026-08-06
updated: 2026-08-06
issue: #31
---

# Handle errors in get_youtube_comments instead of crashing

## Objective

Make `get_youtube_comments` fail gracefully (invalid key, comments
disabled, quota exhaustion) the same way `get_youtube_video_details`
already does, and make `app.py` actually surface either function's error
to the user instead of crashing with a generic 500 or silently rendering
blank output.

## Context

- `src/app_clustering/clustering.py:261-324` — `get_youtube_comments` has
  no try/except, unlike `get_youtube_video_details` (`:228-258`) which
  already returns `{"error": str(e)}` on failure.
- `src/app_clustering/app.py:37-91` — the POST route calls
  `get_youtube_comments` unguarded (`:41`) and never checks
  `video_details.get("error")` either, so even the *existing* error path
  from `get_youtube_video_details` isn't surfaced today
  (`templates/index.html:43-49` renders `video_details.title`
  unconditionally, which would raise `UndefinedError`/render blank if
  `video_details` is an error dict).
- No existing tests for `app_clustering` (`tests/` has none referencing it).
- `flask`/`google-api-python-client` live in the `nlp` Poetry group, but
  `make test` only runs pytest inside `diplomado-core`, which doesn't have
  them installed. There's no `make test-nlp` target, even though
  `build-nlp`/`up-nlp`/`shell-nlp`/`jupyter-nlp` all exist (`Makefile`
  lines 92-106). Adding `test-nlp` (mirroring the existing per-group
  pattern) is required for this story's tests to be runnable at all, and
  is in scope here.

## Requirements

### Functional Requirements

- [ ] `get_youtube_comments` wraps its API calls in try/except, returning
      `{"error": str(e)}` on failure (same shape as
      `get_youtube_video_details`) instead of letting the exception
      propagate
- [ ] `app.py`'s POST handler checks `video_details.get("error")` and
      `comments_df` (post-`get_youtube_comments`) for an error dict/marker
      *before* proceeding to embeddings/clustering, and passes that error
      through to the template instead of continuing the pipeline
- [ ] `templates/index.html` renders the error message when present,
      instead of (or in addition to) the video-details block
- [ ] `Makefile` gets a `test-nlp` target following the existing
      `build-nlp`/`up-nlp`/`shell-nlp` pattern (build image if needed, run
      pytest inside `diplomado-nlp`)

### Non-Functional Requirements

- [ ] No regression to the success path: a valid URL with comments enabled
      renders results exactly as it does today

## Architecture

### Components

- `src/app_clustering/clustering.py` (`get_youtube_comments` gets a
  try/except; needs a way to distinguish "error dict" from "DataFrame" for
  callers — return `{"error": str(e)}` dict on failure, matching
  `get_youtube_video_details`'s existing contract)
- `src/app_clustering/app.py` (`index()` route: check `video_details` and
  the comments-fetch result for an `"error"` key before running the rest
  of the pipeline; pass an `error` value to `render_template`)
- `src/app_clustering/templates/index.html` (new `{% if error %}` block)
- `Makefile` (`test-nlp` target)
- `tests/test_app_clustering.py` (new file, mocks
  `googleapiclient.discovery.build` via `unittest.mock`)

### External Dependencies

No new dependencies — `flask`, `google-api-python-client` already in the
`nlp` group.

## User Stories

Full Gherkin acceptance criteria (5 scenarios: invalid key, comments
disabled, quota exceeded, video-details error, and the happy path) live in
GitHub Issue **#31**.

## Testing Strategy

### Unit/Integration Tests

New `tests/test_app_clustering.py`, run only inside the `nlp` container
(needs `flask`/`google-api-python-client`):

- Mock `googleapiclient.discovery.build` so `get_youtube_comments` raises
  an `HttpError`-like exception (or any `Exception`) → assert it returns
  `{"error": ...}` instead of raising
- Flask test client: POST a URL where the mocked comments fetch returns an
  error → assert the response renders an error message, not a 500 and not
  a blank/partial results page
- Flask test client: POST a URL where the mocked video-details fetch
  returns `{"error": ...}` → assert that error is shown (currently
  silently ignored)
- Flask test client: POST a URL where both mocked calls succeed → assert
  the response still renders sankey/scores/sentiment output (no
  regression)

### Manual Verification

`make test-nlp` runs green.

## Boundaries & Constraints

### In Scope

- Error handling in `get_youtube_comments` and `app.py`'s route
- Minimal template changes to surface the error
- `make test-nlp` Makefile target
- `tests/test_app_clustering.py`

### Out of Scope

- Retrying/backoff on quota errors
- Structured/typed error objects (a plain `{"error": str}` dict matches
  the existing `get_youtube_video_details` convention — no reason to
  diverge)
- Any UI/CSS redesign of `templates/index.html` beyond the error block
- Rate limiting or caching the YouTube API calls

### Technical Constraints

- Tests must run via `docker compose exec diplomado-nlp poetry run pytest`
  (or the new `make test-nlp`), not bare host Poetry

## Success Criteria

- [ ] All 5 Gherkin scenarios from issue #31 have passing automated tests
- [ ] `make test-nlp` target exists and runs green
- [ ] `poetry run pylint $(git ls-files '*.py')` still passes when run
      with the `nlp` group installed (CI's `.github/workflows/pylint.yml`
      installs `--with nlp` specifically because `app_clustering/*.py`
      needs `flask`/`sentence-transformers`/`wordcloud` resolvable for
      pylint's import-error check — `make lint` alone, which only has the
      `core` group, is not sufficient to validate this)
- [ ] Issue #31 closed with evidence

## Implementation Plan

See `specs/youtube-comments-error-handling-plan.md`.
