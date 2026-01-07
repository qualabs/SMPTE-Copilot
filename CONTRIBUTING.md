# Contributing Guide

Thank you for your interest in contributing!
This project aims to build an open-source AI co-pilot for multimedia
archives.

We welcome contributions of all kinds: 
-   code, documentation, ideas, and community support.

------------------------------------------------------------------------

## 🧑‍💻 How to Contribute Code

Follow this standard GitHub workflow:

#### 1. Fork the repository

#### 2. Clone your fork

#### 3. Create a feature branch

-   Examples: 
    -   `feat/video-ingestion-module` 
    -   `fix/metadata-parser-issue`

#### 4. Install pre-commit hooks (recommended)

This project uses [pre-commit](https://pre-commit.com) to automatically run linting checks before commits. Install the hooks:

```bash
# Install development dependencies (includes ruff and pre-commit)
pip install -e ".[dev]"

# Install the git hooks
pre-commit install
```

**Note:** The pre-commit hooks require `ruff` to be installed, which is included in the `[dev]` dependencies. Make sure you've installed the development dependencies before installing the hooks.

After installation, the hooks will automatically run on every commit. You can also run them manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only (automatic on commit)
pre-commit run
```

#### 5. Make your changes
-   Write clear, modular code.

#### 6. Push your branch

#### 7. Open a Pull Request (PR) with: 

- **What** you changed
- **Why** the change is needed
- **How** to test it
- Any **limitations** or **future improvements**

A maintainer will review your PR and provide feedback.

------------------------------------------------------------------------

## 🐛 Reporting Issues

If you find a bug, open an Issue and include: 
- Steps to reproduce
- Expected vs actual behavior
- Logs, screenshots, or example files if relevant

------------------------------------------------------------------------

## 💬 Join the Community on Discord

-   **SMPTE Text Channel:** https://discord.gg/ZcrUp2grg3
-   **SMPTE Voice Channel:** https://discord.gg/VKgFprBQ38
-   **SummerCamp General Channel:** https://discord.gg/cEQN4a2Pjg

------------------------------------------------------------------------

## 🤝 Code of Conduct

Always communicate with professionalism, kindness, and a spirit of collaboration.
Let’s build a positive, respectful, and collaborative community together.

------------------------------------------------------------------------

## 📄 License

By contributing, you agree that your contributions fall under the
project's open-source license.
