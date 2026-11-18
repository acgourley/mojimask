# Contributing to mojimask

Thanks for helping. mojimask is maintained by one person in spare time, so review time is the scarce thing. These
rules exist to spend it well.

## Before you start

- **Open an issue first** for anything bigger than a small fix: a new feature, a change to the scan's rules, a new
  dependency or model. A short "here's what I want to do and why" saves both of us a wasted pull request.
- Set up from source as in [AGENTS.md](AGENTS.md#setup-and-building). Then turn on the privacy check once per clone:

  ```bash
  git config core.hooksPath .githooks
  ```

## Privacy

mojimask exists to protect people in party videos, and the repository follows the same rule: **no real person's
face goes into git, an issue, a pull request or a comment.** That means no clips, frames, crops or screenshots of
real footage. The one exception is the sample clip, `test_clips/jazz.mp4`, a public-domain film from 1928: use it
for screenshots and bug reports. If a bug only shows up on your own footage, describe it in numbers and words
(frame numbers, box sizes, "the second guest from the left") instead.

The pre-commit hook and CI refuse pictures, videos and scan output (`.githooks/check_media.sh`).

## Pull requests

- Keep each one to one change, small enough to review in one sitting.
- Say what changed, why, and how you checked it. Changes to the scan's rules need before and after numbers from
  the rules' benchmark (`mm-bench`, see [AGENTS.md](AGENTS.md#changing-the-scans-rules)), and you should have
  looked at the jazz pictures behind any number that moved.
- Contributions are licensed under the project's licence, AGPL-3.0. You must have the right to submit what you
  send: don't paste code from projects with incompatible licences.

## Using AI coding tools

You're welcome to use them. Most of mojimask itself was written with coding agents, under the maintainer's
direction and review. The repository's [AGENTS.md](AGENTS.md) holds the instructions agents need (Claude Code,
Codex, Cursor, Copilot and others read it on their own).

What we ask in return:

- **You answer for everything you submit.** Read and understand every line before asking for review, and be ready
  to explain any of it in your own words. "The AI wrote it" isn't an answer to a review question.
- **Say so.** Add an `Assisted-by:` trailer to each commit where a tool wrote a substantial part, naming the tool
  and model, for example `Assisted-by: Claude Code (claude-opus-5-5)`, and tick the box in the pull request
  template. This helps reviewers know where to look. It isn't held against you.
- **Write the pull request description, issues and review replies yourself.** Use a tool to fix the grammar or
  translate if you like, but the words should be yours.
- **No unattended agents.** An agent may not open issues, pull requests or comments, or push, without a person
  reading what it wrote first.
- Pull requests that would cost more to review than they're worth (large, unexplained, unrequested, or clearly
  unread) may be closed with a short note and no detailed review. A smaller, explained version is welcome.
