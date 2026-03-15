"""
Claude Code Review Script
-------------------------
This script runs inside GitHub Actions.
It finds changed Python files and asks Claude to review them.
"""

import os
import subprocess
import anthropic


def get_changed_files():
    """Get list of Python files changed in the last commit."""
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--name-only", "--diff-filter=AM"],
        capture_output=True,
        text=True,
    )
    files = result.stdout.strip().split("\n")
    # Only review Python files
    return [f for f in files if f.endswith(".py") and os.path.exists(f)]


def read_file(path):
    """Read file contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def review_file(client, filename, code):
    """Ask Claude to review a single file."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are a helpful code reviewer. "
            "Review the provided Python code and give beginner-friendly feedback. "
            "Focus on: 1) correctness, 2) readability, 3) any bugs or improvements. "
            "Keep feedback concise and constructive. Use simple language."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Please review this Python file: {filename}\n\n```python\n{code}\n```",
            }
        ],
    )
    return response.content[0].text


def main():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    print("=" * 55)
    print("       Claude Code Review")
    print("=" * 55)

    changed_files = get_changed_files()

    if not changed_files:
        print("No Python files changed in this push. Nothing to review.")
        return

    print(f"\nFound {len(changed_files)} changed Python file(s):\n")

    for filename in changed_files:
        print(f"Reviewing: {filename}")
        print("-" * 55)

        code = read_file(filename)
        feedback = review_file(client, filename, code)

        print(feedback)
        print()

    print("=" * 55)
    print("Review complete!")


if __name__ == "__main__":
    main()
