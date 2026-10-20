# Liminal project rules

## Objective and design

- Build **Liminal**, an original spiking neural network (SNN), from scratch in
  Rust, including its training and inference workflows.
- The primary goal is **natural language generation**. Design the model and
  learning algorithms around causal next-token prediction and autoregressive
  text generation, with explicit mathematical definitions and measured results.
- Use **FineWeb** (`HuggingFaceFW/fineweb`) as the primary training corpus.
  Download and prepare it in Rust with explicit size budgets, source provenance,
  hashes, and disjoint whole-document train/validation splits. Keep document
  boundaries during training/evaluation; do not mix validation into vocabulary
  fitting or updates. Synthetic data is for focused tests only.
- Design and implement Liminal's architecture directly. Do not base it on a port,
  fork, or renamed copy of another model's codebase.
- There is no migration or external-model parity prerequisite. Develop Liminal
  against explicit design goals and reproducible measurements of its own behavior.
- Document neuron dynamics, spike representation, state/reset behavior, the
  learning method, and input/output behavior as the architecture is developed.

## Project boundaries

- This directory is the Liminal project root. Keep Liminal source, tests,
  configuration, build output, data, checkpoints, and generated artifacts within
  this project and organized by purpose.
- Do not copy, move, vendor, symlink, or hardlink files between Liminal and other
  model projects. Independently implement the model and its learning behavior.
- Do not introduce another model checkout as a runtime or training dependency.
  Research and published equations may inform documented design decisions;
  implementation remains Liminal's own.

## Rust only

- Use **Burn** for tensor execution, automatic differentiation, optimization, and
  batched GPU training. WGPU is the default device; CPU checks run the same Burn
  implementation. Preserve the specified hard spikes, triangular surrogate, and
  attached reset gradient with mathematical and numerical tests.
- Keep one model and training implementation across devices. Do not retain the
  former scalar forward/backward/optimizer as an alternate production path.
- All Liminal executable implementation must be Rust, including training,
  inference, tokenization, data preparation, evaluation, tests, checkpoint tools,
  and project utilities.
- Do not add Python helpers, Python training bridges, or project-owned C/C++/CUDA
  kernels. Implement model operations and any required gradients in Rust.
- Choose Rust-native dependencies. Do not use a non-Rust ML runtime as a shortcut
  around the Rust-only requirement.
- Markdown, Cargo manifests, configuration, datasets, and model artifacts are
  allowed. Shell commands for inspection and standard development tools do not
  replace Rust implementation.

## Organization and one maintained implementation

- Use a coherent Cargo project or workspace. Keep modules focused, with clear
  boundaries between model operations, neurons/state, data/tokenization, training,
  inference, evaluation, and persistence as those components are implemented.
- Keep entry points thin; put shared behavior in library modules. Add structure
  when it has a real responsibility, without creating empty placeholder modules.
- Maintain a single current implementation. Edit and replace code in place; do
  not create parallel `v1`/`v2`, `old`, `legacy`, backup, or experimental code trees.
- Remove obsolete implementations, dead code, abandoned dependencies, and stale
  configuration as part of the change that supersedes them. Do not retain
  commented-out implementations.
- This single-version rule concerns code organization; it does not require
  deleting Git history or omitting Cargo's package metadata.
- Keep generated files out of source modules. Keep tests organized by the behavior
  they verify and avoid unnecessary duplicate implementations in test helpers.

## Validation and evidence

- Define what each model or training change should demonstrate before judging
  its results. Record input/data specifications, splits, configuration, seeds,
  precision, learning settings, compute/training budget, metrics, and numerical
  tolerances when relevant.
- Verify neuron dynamics, spike/state/reset behavior, learning updates, and
  end-to-end training and inference as those components are implemented. Test
  gradients when the selected learning method uses them.
- Record commands, environment, actual measurements, limitations, and unverified
  areas under `docs/`. Successful compilation or plausible output alone does not
  establish model quality or learning capability.
- Keep claims tied to evidence. If data or compute prevents validation, document
  the gap instead of reporting an unverified result as achieved.

## Documentation and experiments

- Keep a `docs/` directory for architecture, design decisions, reproduction
  procedures, and validation evidence.
- Use `docs/experiments/` to log experiments from the start of development.
- Each experiment should record its question/hypothesis, configuration and seed,
  data and compute budget, commands, metrics and any established Liminal baseline,
  outcome, and resulting decision. Keep the experiment's code in the single
  maintained implementation and remove approaches that are superseded or rejected.

## Current library documentation

<!-- context7 -->
Use Context7 MCP to fetch current documentation whenever the user asks about a
library, framework, SDK, API, CLI tool, or cloud service -- even well-known ones.
This includes API syntax, configuration, version migration, library-specific
debugging, setup instructions, and CLI tool usage. Use even when you think you know
the answer. Prefer this over web search for library docs.

Do not use for refactoring, writing scripts from scratch, debugging business logic,
code review, or general programming concepts.

1. Always start with `resolve-library-id` using the library name and the user's
   question, unless the user provides an exact library ID in `/org/project` format.
2. Pick the best match by exact name, description relevance, code snippet count,
   source reputation (High/Medium preferred), and benchmark score. If results do
   not fit, try alternate names or queries. Use version-specific IDs when the user
   mentions a version.
3. Call `query-docs` with the selected library ID and the user's full question,
   rather than single words.
4. Base the answer on the fetched documentation.
<!-- context7 -->
