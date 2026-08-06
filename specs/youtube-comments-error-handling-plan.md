# Implementation Plan: Handle errors in get_youtube_comments

**Spec**: `specs/youtube-comments-error-handling.md`
**Created**: 2026-08-06
**Status**: approved

## Components

### 1. `test-nlp` Makefile target
- **Purpose**: Make it possible to actually run the tests this story adds
- **Files**: `Makefile`
- **Effort**: XS

### 2. `get_youtube_comments` error handling
- **Purpose**: Return `{"error": str(e)}` instead of raising
- **Files**: `src/app_clustering/clustering.py`
- **Effort**: XS

### 3. `app.py` route error surfacing
- **Purpose**: Check both fetches for an error dict before running the
  rest of the pipeline; pass error to template
- **Files**: `src/app_clustering/app.py`
- **Effort**: S

### 4. Template error block
- **Purpose**: Render the error message to the user
- **Files**: `src/app_clustering/templates/index.html`
- **Effort**: XS

### 5. Test suite
- **Purpose**: Cover all 5 Gherkin scenarios from issue #31
- **Files**: `tests/test_app_clustering.py` (new)
- **Effort**: M

## Dependencies

### Build Order

1. `test-nlp` Makefile target first (needed to run/verify everything else)
2. `get_youtube_comments` try/except (foundation — app.py depends on its
   new return contract)
3. `app.py` route changes (depends on step 2's error-dict contract)
4. Template error block (depends on `app.py` passing `error` to the
   template context)
5. Tests (can be written test-first per component, but full suite only
   runnable end-to-end once 2-4 are in place)

### External Dependencies

None new.

## Risks & Assumptions

### Risks

- **Risk**: mocking `googleapiclient.discovery.build` for pagination
  (`get_youtube_comments` loops on `nextPageToken`) is fiddly — the mock
  needs to support both the initial call and the paginated follow-up call
  args. **Mitigation**: keep the success-path test's mocked response
  single-page (no `nextPageToken` key) to sidestep this; only the
  error-path tests need the mock to raise, which happens before pagination
  logic runs anyway.
- **Risk**: `app.py`'s current flow computes `comments_df` via a chain of
  several `clustering.*` calls after the fetch — inserting an early return
  needs care not to skip cleanup/leave partial state. **Mitigation**: the
  route already returns a single `render_template(...)` at the end; an
  early error just needs to set `error` and skip straight to that same
  return (Python's normal control flow, no new exit points needed).

### Assumptions

- Flask's default `render_template` behavior with `video_details.title`
  where `video_details` is `{"error": ...}` today either raises
  `jinja2.exceptions.UndefinedError` (accessing `.title` on a dict without
  that key, depending on Jinja's attribute-vs-item fallback for dicts) or
  silently renders empty — either way, current behavior is broken; adding
  the explicit `error` check fixes both.

## Milestones

- [ ] Milestone 1: `make test-nlp` runs (even with 0 test files) proving
      the target itself works
- [ ] Milestone 2: all 5 Gherkin scenarios pass
- [ ] Milestone 3: `poetry run pylint` (with `nlp` group) still clean

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Add `test-nlp` Makefile target
  - **Acceptance**: `make test-nlp` builds/starts `diplomado-nlp` and runs
    `poetry run pytest tests -v` inside it, mirroring `test:`'s pattern
    but targeting the `nlp` service
  - **Files**: `Makefile`
  - **Tests**: manual run against the existing (empty) test file for
    app_clustering, then against the new one once added
  - **Effort**: XS

- [ ] **Task 2**: `get_youtube_comments` try/except
  - **Acceptance**: on any exception during the API calls, returns
    `{"error": str(e)}`; success path (a DataFrame) unchanged
  - **Files**: `src/app_clustering/clustering.py`
  - **Tests**: covered by Task 5's suite
  - **Effort**: XS

### Features (Build Second)

- [ ] **Task 3**: `app.py` route error surfacing
  - **Acceptance**: if `video_details` has an `"error"` key, or the
    comments fetch returns `{"error": ...}` (dict, not DataFrame), the
    route skips embeddings/clustering and renders the template with an
    `error` value set; happy path passes `error=None`
  - **Files**: `src/app_clustering/app.py`
  - **Tests**: covered by Task 5's suite
  - **Effort**: S

- [ ] **Task 4**: Template error block
  - **Acceptance**: `{% if error %}` block renders the message;
    `video_details` block only renders when there's no error
  - **Files**: `src/app_clustering/templates/index.html`
  - **Tests**: covered by Task 5's Flask-test-client assertions (checking
    rendered HTML contains the error text)
  - **Effort**: XS

### Integration (Build Third)

- [ ] **Task 5**: `tests/test_app_clustering.py`
  - **Acceptance**: 5 tests, one per Gherkin scenario in issue #31, all
    passing under `make test-nlp`
  - **Files**: `tests/test_app_clustering.py`
  - **Tests**: itself
  - **Effort**: M

## Effort Estimate

**Total Estimated Days**: ~0.3 day

| Phase | Effort |
|-------|--------|
| Foundation | 0.05 day |
| Features | 0.1 day |
| Integration | 0.15 day |
