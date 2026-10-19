# Documentation research notes

This note records the documentation practices used to reorganize BioSynth's project docs. It is a short reference for future doc changes, not a substitute for the project-specific guides.

## Findings from primary sources

- GitHub recommends that a repository README explain what the project does, why it is useful, how to get started, and where to find help. GitHub also advises keeping the README focused on information needed to get started and contribute, with longer material in dedicated documentation. [About the repository README file](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- GitHub's documentation guidance recommends choosing content around reader goals, using meaningful headings, putting important context first, and writing concise, active sentences. [Best practices for GitHub Docs](https://docs.github.com/en/contributing/writing-for-github-docs/best-practices-for-github-docs)
- Diátaxis groups documentation by reader need: tutorials help people learn, how-to guides help them complete a task, reference material describes interfaces and facts, and explanations build understanding. [Diátaxis: Start here](https://diataxis.fr/start-here/)
- Google's technical-writing material recommends short sentences that focus on one idea and generally prefers active voice. [Short sentences](https://developers.google.com/tech-writing/one/short-sentences), [Active voice](https://developers.google.com/tech-writing/one/active-voice)

## Application to BioSynth

- Keep the root README short: purpose, safety boundary, setup, launch commands, tests, and a link to the docs index.
- Put each task or technical subject in its own document. Make the docs index the route into those pages.
- Separate the two runnable Streamlit applications in every relevant setup reference: root `app.py` is the model-backed event replay; `ui/app.py` is the command-center interface.
- Treat source, configuration, generated metadata, and reports as authoritative. Mark generated artifacts as generated, and label unverified or synthetic results clearly.
- Keep the POC disclaimer and the distinction between Hall 2018 evidence and synthetic fusion results close to any model-performance claims.
- Prefer relative links so docs also work in a local clone.

## Maintenance rule

When changing a documented behavior, update the owning reference first and then update links or short summaries elsewhere. Avoid copying long feature lists, command inventories, or metric tables into multiple documents. Verify every command and path against the current repository before publishing a doc edit.
