"""
Claude Code Review Script
-------------------------
Runs inside GitHub Actions on Pull Requests.
Reviews changed Python files and posts feedback as a PR comment.
"""

import os
import subprocess
import anthropic
import requests


def get_changed_files():
    """Get list of Python files changed in this PR."""
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--name-only", "--diff-filter=AM"],
        capture_output=True,
        text=True,
    )
    files = result.stdout.strip().split("\n")
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
            "Format your response in markdown so it looks good in GitHub comments."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Please review this Python file: `{filename}`\n\n```python\n{code}\n```",
            }
        ],
    )
    return response.content[0].text


def post_pr_comment(comment_body):
    """Post a comment on the Pull Request using GitHub API."""
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO"]
    pr_number = os.environ["PR_NUMBER"]

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.post(url, headers=headers, json={"body": comment_body})

    if response.status_code == 201:
        print(f"Comment posted successfully: {response.json()['html_url']}")
    else:
        print(f"Failed to post comment: {response.status_code} {response.text}")


def main():
    client = anthropic.Anthropic()

    print("=" * 55)
    print("       Claude Code Review")
    print("=" * 55)

    changed_files = get_changed_files()

    if not changed_files:
        print("No Python files changed. Nothing to review.")
        # Post a comment saying nothing changed
        post_pr_comment("### Claude Code Review\n\nNo Python files were changed in this PR.")
        return

    print(f"\nFound {len(changed_files)} changed Python file(s): {changed_files}\n")

    # Build the full comment with reviews for all changed files
    comment_parts = ["## 🤖 Claude Code Review\n"]
    comment_parts.append(f"Reviewed **{len(changed_files)}** Python file(s):\n")

    for filename in changed_files:
        print(f"Reviewing: {filename} ...")
        code = read_file(filename)
        feedback = review_file(client, filename, code)

        comment_parts.append(f"---\n### 📄 `{filename}`\n")
        comment_parts.append(feedback)
        comment_parts.append("\n")

    comment_parts.append("---\n*Reviewed by Claude via GitHub Actions*")

    # Combine all parts into one comment
    full_comment = "\n".join(comment_parts)

    # Print to logs
    print(full_comment)

    # Post to PR
    print("\nPosting comment to PR...")
    post_pr_comment(full_comment)


if __name__ == "__main__":
    main()
