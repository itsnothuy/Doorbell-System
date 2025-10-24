
title: "ADR 004: Testing Strategy" date: 2025-10-23 status: accepted

Context
Frigate's upstream project has historically relied on integration and community testing rather than an extensive unit test suite. Due to the nature of processing live video streams, much of the behaviour is best verified in real environments with a variety of camera models, accelerators and network conditions. However, for our integration work and ongoing maintenance, we need a more structured testing approach.
Decision
We adopt a layered testing strategy comprising:
1. Unit tests for pure functions and configurable logic. Where possible, functions in the integration (e.g. configuration parsers, threshold logic, provider selection) should be unit tested using pytest. Unit tests should avoid external dependencies and run quickly.
2. Integration tests for end‑to‑end pipelines. We will build a set of small test fixtures that simulate camera streams (for example, prerecorded video clips) and verify that the pipeline produces expected events, recordings and metadata. These tests may use temporary directories and a local database. When generative AI is enabled, tests should mock the provider to avoid network calls.
3. Manual and real‑world testing for hardware‑specific scenarios. Certain features (e.g. EdgeTPU acceleration or specific camera protocols) cannot be fully simulated in code. We will document manual test procedures and encourage testers to run the system on representative hardware.
4. Continuous integration (CI). All PRs must run the test suite and ensure that builds succeed. Where possible, CI should include basic smoke tests for the main pipeline using a short sample video.
Rationale
* Risk mitigation. Unit tests catch regressions in logic and configuration parsing early. Integration tests ensure that the pipeline works across module boundaries. Manual testing covers environments that automated tests cannot easily emulate.
* Maintainability. A layered approach allows us to expand the integration in the future while confidently refactoring internals.
* Community involvement. Leveraging real‑world testers (as the upstream project does) is still important. Clear documentation and sample configurations will facilitate community validation.
Consequences
* Additional effort is required to create and maintain test fixtures (sample videos, mock providers). When adding new features, developers must consider how to test them.
* CI runs may take longer due to integration tests. We will strive to keep the test suite efficient by keeping sample videos short and parallelising test execution where feasible.
* Some hardware‑specific behaviours may remain untested in CI.

