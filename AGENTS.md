
## Persistent Codex handoff

Maintain repository-level continuity across Codex sessions.

### At the start of every task
- Read docs/codex/STATE.md.
- Read the latest relevant entries in docs/codex/LOG.md.
- Check Git status and the current branch.
- Use these files to reconstruct context, but verify claims
  against the actual repository before relying on them.

### During work
- Update docs/codex/STATE.md after meaningful milestones.
- Keep the state concise and relevant to future sessions.
- Append important decisions, experiments, and verified
  results to docs/codex/LOG.md.
- Record relevant files, commands, test outcomes, and
  commit hashes when available.
- Distinguish completed work from unverified hypotheses.

### Before ending a session
- Update STATE.md with:
  - Current objective
  - Completed work
  - Verified results and test status
  - Important decisions and assumptions
  - Uncommitted changes
  - Remaining blockers
  - Specific next actions
- Add a dated summary to LOG.md.
- Provide a concise human-readable handoff summary.

### Safety and Git
- Never store credentials, tokens, passwords, or private
  subject information in checkpoint documents.
- Commit and push whenever significant progress has been made
- Do not overwrite or discard unrelated changes.
- Do not claim a task or test succeeded without verification.
- When multiple agents work concurrently, avoid conflicting
  edits to shared checkpoint documents.
