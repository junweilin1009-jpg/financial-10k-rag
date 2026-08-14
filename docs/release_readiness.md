# Release Readiness Review

Review date: 2026-08-14
Branch reviewed: `codex/refactor-p0-foundation`

## Outcome

The repository is locally portfolio-ready from a code, packaging, testing, documentation, and Git
history perspective. Public open-source release is **not yet approved** because the team has not
selected a software license or confirmed redistribution terms for the course-provided PDFs.
Remote CI and billable model evaluation also remain intentionally unverified.

## Verified locally

### Code and architecture

- [x] Core logic lives in the installable `src/financial_rag` package.
- [x] Streamlit, CLI, Colab, and evaluation use the shared package rather than separate engines.
- [x] Cache, response normalization, schemas, filing validation, configuration, and logging have
      explicit module boundaries.
- [x] Public build, answer, and source result structures are typed.
- [x] Configuration rejects invalid chunk, retrieval, output, and model settings before execution.
- [x] No tracked source or config file contains a developer-specific macOS or Windows home path.

### Data and evidence correctness

- [x] Exactly one supported 2025 Form 10-K is required for each issuer.
- [x] Issuer, form type, and fiscal period are validated from PDF content.
- [x] The three bundled files match the byte sizes and SHA-256 values in `data/manifest.csv`.
- [x] Model-facing sources are restricted to documents that fit into the actual context.
- [x] Table and full-page enrichment failures log a warning and preserve a documented fallback.

### Security and privacy

- [x] No key-shaped `sk-...` credential was found in tracked text files.
- [x] `.env`, caches, outputs, virtual environments, build artifacts, and Python bytecode are ignored.
- [x] Streamlit passes its key directly to clients instead of mutating process-wide environment state.
- [x] FAISS persistence uses a native index plus validated JSON mapping, not executable pickle loading.

### Reproducibility and packaging

- [x] Python support is declared as 3.11-3.13.
- [x] `pyproject.toml` is the single direct-dependency and packaging source.
- [x] A clean Python 3.12 environment installed the project and development dependencies.
- [x] `pip check` reported no broken requirements.
- [x] The installed `financial-rag-chat --help` command completed successfully.
- [x] The Colab notebook parses as JSON and contains no saved execution output.
- [x] All documented relative Markdown links resolve locally.

### Tests and automation

- [x] 42 offline tests pass without an API key.
- [x] Tests include unit, failure-path, safe-cache, corpus-integrity, holdout, and answer-workflow
      integration coverage.
- [x] Ruff lint and format checks pass.
- [x] GitHub Actions is configured to run install, dependency validation, Ruff, tests, and CLI smoke
      checks on Python 3.11, 3.12, and 3.13.
- [x] Holdout execution requires explicit acknowledgement and a clean committed worktree.
- [x] Evaluation exports record UTC timestamp, commit SHA, and worktree state.

### Portfolio documentation

- [x] README explains the problem, features, setup, usage, evaluation, results, limitations, and team.
- [x] Architecture documentation covers data flow, component ownership, failure policy, and trade-offs.
- [x] Methodology documentation separates model experiments from retrieval improvements.
- [x] A Chinese interview guide provides 30-second, 1/3/5-minute narratives, follow-up answers, a
      STAR story, and evidence-based resume bullets.
- [x] Recorded benchmark figures retain their small-sample, post-tuning, date, and pricing caveats.

## Required before public open-source release

- [ ] Confirm that the three course-provided PDFs may be redistributed in a public repository.
- [ ] Select a software license with all team members; do not assume MIT consent.
- [ ] If PDF redistribution is not allowed, remove the files, retain metadata/source instructions,
      and provide a documented download/setup step.
- [ ] Add a Git remote, push the branch, and confirm the real GitHub Actions matrix succeeds.

## Required before claiming post-refactor model quality

- [ ] Use an authorized API key to run a small end-to-end embedding and answer smoke test.
- [ ] Freeze and commit the code before selecting the protected holdout.
- [ ] Run the 15-question holdout once, preserve all failures, and report it separately.
- [ ] Do not tune against the holdout after seeing its answers.

These items require team authorization, external service access, or billable calls and therefore
were not inferred from the refactoring request.

## Known maintenance item

`langchain-community` currently emits a sunset/deprecation warning for its FAISS integration. The
current pinned dependency remains functional and all tests pass. Migration should be a separate,
tested change to a maintained standalone integration rather than an unreviewed dependency swap in
the release-polish commit.
