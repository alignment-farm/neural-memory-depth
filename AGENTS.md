# Neural memory depth

Read [README.md](README.md) first. It owns the question, initial expectation,
starting references, and links to this study's eventual findings.

This project investigates the operating conditions behind the different
neural-memory depth trends reported in Titans and Modular TTT. Its first
intended output is a written mechanism and configuration comparison. The root
prepared the directory without starting the investigation; the ancillary
agent owns the work in its own session.

The study owns its methods, analysis, supporting evidence, and local publication.
Revise the question when the sources warrant it and explain consequential
changes. An explanation that resolves the apparent tension is useful progress.
The parent's S3 sketch supplies background, not a mandatory experimental
protocol. Initial scope and resource expectations are in the README.

## Research practice

- Inspect the closest papers and available implementations. Record exact paper
  versions, sections, code revisions, and configurations behind substantive
  claims; distinguish reported results, interpretation, and reproduction.
- Establish comparability before attributing differences to depth. Treat
  parameter matching and compute matching as distinct comparisons, and mark
  missing information rather than filling it with assumptions.
- Keep notes proportional to the question. Publish findings and limitations
  locally, linked from the README. A new experiment or positive depth effect
  is not required for a useful contribution.
- For arXiv discovery and metadata, use `https://export.arxiv.org/api/query`.
  Cache responses, use a descriptive User-Agent, and use one connection with
  at least three seconds between API/OAI requests across clients under your
  control; honor `Retry-After`. Retrieve versioned paper content separately.
  The [parent's source guidance](../../construct-2/AGENTS.md#research-sources)
  supplies the access details.

## Model resources

- Dedicated Mac Studio M1 (64 GB unified memory), serving over Tailscale through
  Docker Model Runner (preferred).
  Chat-completions endpoint:
  `https://mac-studio-7hr7.taile71f88.ts.net/engines/v1/chat/completions`
- Local open-weight models with `docker model`
- OpenAI models with `codex`
- SpaceXAI models with `agent`

Verify model revisions and gradient or mutable-state access on the chosen
resource before relying on it for a neural intervention. The endpoint is
carried forward from the lab's resource instructions; project preparation
did not test its availability or training capabilities.

## Dependency management

- Use `uv` for Python package and project management.
- Use Docker and Compose/Dockerfiles for supporting resources when needed.
