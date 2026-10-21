# Workflows

Each recipe assumes `nalcos --json status` for the same scope reported a usable index. Add
`--freshness cached` to every search when the environment must not write files.

## Find the commit that introduced a behavior

1. Search with a description of the change and a short result list.

   ```sh
   nalcos --json search "retry requests after a connection reset" --limit 5 --timeout 60s
   ```

2. Read the evidence of the top results. Expand any clipped excerpt.

   ```sh
   nalcos --json show --evidence EVIDENCE_ID
   ```

3. If nothing fits, rephrase with the identifiers, config keys, or error strings the code
   would contain, and try `--mode lexical` with those names.

   ```sh
   nalcos --json search "max_retries backoff_factor ConnectionResetError" --mode lexical --limit 5
   ```

4. Confirm the winner with Git. Use `git log -S` once you know an exact string, to find the
   first commit that added it.

   ```sh
   git show COMMIT_SHA
   git log -S'backoff_factor' --oneline --reverse
   ```

## Hunt for a regression between two revisions

Use a known-good revision `A` and a known-bad revision `B`. `--range A..B` covers commits
reachable from `B` but not from `A`, which are exactly the changes that could have caused it.

1. Check that the range is indexed; finish indexing if it is not.

   ```sh
   nalcos --json status --range v1.4.0..v1.5.0
   nalcos --json sync --range v1.4.0..v1.5.0 --timeout 10m
   ```

2. Search for the symptom and for the likely cause, separately.

   ```sh
   nalcos --json search "uploads fail with a timeout" --range v1.4.0..v1.5.0 --limit 10 --timeout 60s
   nalcos --json search "change upload chunk size or HTTP client timeout" --range v1.4.0..v1.5.0 --limit 10 --timeout 60s
   ```

3. Treat every result as a candidate. Read each patch, then confirm with a test or a build at
   the candidate and at its parent.

   ```sh
   git show CANDIDATE_SHA
   git log --oneline v1.4.0..v1.5.0 -- src/upload
   ```

4. When candidates do not settle it, ask the user before running `git bisect`, because it
   checks out other commits in the working tree. nalcos does not run bisects.

   ```sh
   git bisect start v1.5.0 v1.4.0
   ```

Indexed diffs compare each commit with its first parent, so a merge commit's evidence covers
everything its branch brought in. Add `--first-parent` to search only mainline commits and
merges, which maps a regression to the merge that introduced it; then search again inside that
merge's branch with `--range MERGE_SHA^1..MERGE_SHA^2`.

## Explain why a file or module changed

Restrict evidence to a pathspec and ask about intent. Commit messages qualify when the commit's
first-parent changes touch a matching path.

```sh
nalcos --json search "why the cache switched to write-through" --path 'src/cache/**' --limit 5 --timeout 60s
nalcos --json show COMMIT_SHA --path 'src/cache/**' --context 5
```

For a full chronology of one file, use Git directly:

```sh
git log --follow --oneline -- src/cache/store.rs
```

## Narrow by people, time, or branches

```sh
nalcos --json search "switch to structured logging" --author alice@example.com --since 2024-01-01 --until 2024-03-31
nalcos --json search "hotfix for duplicate invoices" --ref release/2.3 --ref main --limit 5
nalcos --json search "temporary debug flag" --all-refs --freshness cached
```

- `--author` matches one case-insensitive substring of `Name <email>`. Run separate searches
  for several authors.
- Dates are committer dates. A rebased or cherry-picked commit has a later committer date than
  its author date (`authored_at` in the JSON).
- `--all-refs` includes local branches, tags, worktree HEADs, and `refs/stash`, but only refs
  that exist locally.

## Search another checkout

```sh
nalcos --repo ../service --json status
nalcos --repo ../service --json search "rate limit headers" --limit 5 --timeout 60s
```

Linked worktrees of one repository share an index; each still searches its own `HEAD` unless
you pass a scope option.

## Prove that something never happened

nalcos can only support absence, never prove it. Before saying a change does not exist:

1. Search the widest relevant scope (`--all-refs`) with `--freshness wait --timeout 5m`.
2. Confirm `history_coverage.complete` is `true` and `warnings` is empty.
3. Run `nalcos --json omissions --all-refs` and check that no omitted path could hold the change.
4. Try at least two different wordings and one lexical search with exact identifiers.
5. Cross-check with `git log -S'IDENTIFIER' --all` or `git log -G'PATTERN' --all`.

Report the scope, the queries, and the coverage you observed.
