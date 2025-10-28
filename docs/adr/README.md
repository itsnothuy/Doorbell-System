# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Doorbell Security System. ADRs document important architectural decisions, their context, rationale, and consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./ADR-001-system-architecture.md) | System Architecture - Modular Monolith | Accepted | 2025-10-23 |
| [ADR-002](./ADR-002-face-recognition-implementation.md) | Face Recognition Implementation | Accepted | 2025-10-23 |
| [ADR-003](./ADR-003-generative-ai-integration.md) | Generative AI Integration | Accepted | 2025-10-23 |
| [ADR-004](./ADR-004-testing-strategy.md) | Testing Strategy | Accepted | 2025-10-23 |
| [ADR-005](./ADR-005-pipeline-architecture-orchestration.md) | Pipeline Architecture and Orchestration | Accepted | 2025-10-28 |
| [ADR-006](./ADR-006-communication-infrastructure-message-bus.md) | Communication Infrastructure and Message Bus | Accepted | 2025-10-28 |
| [ADR-007](./ADR-007-detector-strategy-hardware-abstraction.md) | Detector Strategy Pattern and Hardware Abstraction | Accepted | 2025-10-28 |
| [ADR-008](./ADR-008-production-deployment-infrastructure.md) | Production Deployment and Infrastructure | Accepted | 2025-10-28 |
| [ADR-009](./ADR-009-security-architecture-compliance.md) | Security Architecture and Compliance | Accepted | 2025-10-28 |
| [ADR-010](./ADR-010-storage-data-management.md) | Storage and Data Management | Accepted | 2025-10-28 |

## ADR Template

When creating new ADRs, use this template:

```markdown
# ADR XXX: [Title]

**Date:** YYYY-MM-DD  
**Status:** [Proposed | Accepted | Deprecated | Superseded]

## Context

Describe the architectural issue or decision that needs to be made.

## Decision

State the architectural decision and explain the reasoning.

## Alternatives Considered

List other options that were considered and why they were rejected.

## Consequences

Describe the positive and negative consequences of this decision.
```

## How to Use ADRs

1. **Before major architectural changes**: Create an ADR to document the decision-making process
2. **During code reviews**: Reference relevant ADRs to ensure consistency
3. **For new contributors**: Read ADRs to understand architectural decisions
4. **When questioning decisions**: Check if an ADR exists explaining the rationale

## ADR Workflow

1. **Propose**: Create a new ADR with status "Proposed"
2. **Review**: Discuss in GitHub issues or PRs
3. **Accept**: Change status to "Accepted" when consensus is reached
4. **Supersede**: When decisions change, mark old ADRs as "Superseded" and create new ones