import os
import requests
import typing
import math
from datetime import timezone
from git import Repo

from utils import get_owner_and_repo

__all__ = ["get_local_commits", "get_github_commits"]


def get_local_commits(
    location: typing.Union[str, os.PathLike],
    max_count: int = 1000,
    branch: typing.Optional[str] = None,
) -> typing.List[typing.Dict[str, str]]:
    """
    Gets the commits from a local repository.

    Arguments
    ---------
    location: str or PathLike
        The location of the repository.
    max_count: int
        The maximum number of commits to return. Defaults to 1000.
    branch: str
        The branch to get the commits from. If not specified, the current branch is used.

    Returns
    -------
    commits: typing.List[typing.Dict[str, str]]
        A list of commits.
    """
    repo = Repo(location)
    # If branch is not specified, use the current branch where the HEAD is at.
    if branch is None:
        branch = repo.head.ref.name

    commits = []

    try:
        for commit in repo.iter_commits(branch, max_count=max_count):
            commits.append(
                {
                    "author": str(commit.author),
                    "email": commit.author.email,
                    "message": commit.message.strip("\n"),
                    "id": commit.hexsha,
                    # Convert datetime to ISO 8601 format.
                    "commit_date": commit.committed_datetime.astimezone(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "branch": branch,
                }
            )
    # If exception is raised, return an empty list.
    except:
        pass

    return commits


def get_github_commits(
    location: str, max_count: int = 1000, branch: typing.Optional[str] = None
) -> typing.List[typing.Dict[str, str]]:
    """
    Gets the commits from a GitHub repository.

    Arguments
    ---------
    location: str
        The location of the repository.
    max_count: int
        The maximum number of commits to return. Defaults to 1000.
    branch: str
        The branch to get the commits from. If not specified, the default branch is used.

    Returns
    -------
    commits: typing.List[typing.Dict[str, str]]
        A list of commits.
    """
    owner, repo = get_owner_and_repo(location)

    # If branch is not specified, use the default branch.
    if branch is None:
        response = requests.get(f"https://api.github.com/repos/{owner}/{repo}").json()
        branch = response["default_branch"]

    # Get a list of all branches in the repository to check if the specified branch exists.
    all_branches_response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/branches"
    ).json()
    all_branches_names = [
        remote_branch["name"] for remote_branch in all_branches_response
    ]

    commits = []

    # Retrieve commits if branch exists in the list of branch name, else return an empty list.
    if branch in all_branches_names:
        # Branch hash to get commits form that branch using the GitHub API.
        branch_hash = all_branches_response[all_branches_names.index(branch)]["commit"][
            "sha"
        ]

        # GitHub API returns a maximum of 100 commits per page. So, we retrieve commits across several pages.
        num_pages = int(math.ceil(max_count / 100))
        # Variable to iterate over the pages.
        i = 1
        run_get_commits_loop = True

        try:
            while run_get_commits_loop:
                commits_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch_hash}&per_page=100?page={i}"
                ).json()
                # If we don't get 100 commits in a page, we know we have reached the last commit.
                if len(commits_response) < 100 or i == num_pages:
                    run_get_commits_loop = False
                for commit in commits_response:
                    commits.append(
                        {
                            "author": commit["commit"]["author"]["name"],
                            "email": commit["commit"]["author"]["email"],
                            "message": commit["commit"]["message"].strip("\n"),
                            "id": commit["sha"],
                            "commit_date": commit["commit"]["author"]["date"],
                            "branch": branch,
                        }
                    )
                i += 1
            # Only get the first max_count commits.
            commits = commits[:max_count]
        # If exception is raised, return an empty list.
        except:
            pass

    return commits
