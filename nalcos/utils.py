import os
import re
import typing
import requests
from git import Repo

__all__ = [
    "get_owner_and_repo",
    "is_local_git_repo",
    "is_github_repo",
    "get_type_of_location",
]


def get_owner_and_repo(location: str) -> typing.Tuple[str, str]:
    """
    Get the owner and repo name from a location string.

    Parameters
    ----------
    location : str
        The location string.

    Returns
    -------
    owner : str
        The owner of the repo.
    repo : str
        The repo name.
    """
    owner, repo = location.strip("/").split("/")
    return (owner, repo)


def is_local_git_repo(location: typing.Union[str, os.PathLike]) -> bool:
    """
    Check if a location is a local git repo.

    Parameters
    ----------
    location : str or os.PathLike
        The location to check.

    Returns
    -------
    bool
        True if the location is a local git repo, False otherwise.
    """
    try:
        # If Repo(location) returns an exception, it means the location does not exists, or is not a git repo
        _ = Repo(location)
        return True
    except Exception:
        return False


def is_github_repo(location: str) -> bool:
    """
    Check if a location is a GitHub repo.

    Parameters
    ----------
    location : str
        The location to check.

    Returns
    -------
    bool
        True if the location is a GitHub repo, False otherwise.
    """
    # Check if a string is of the type '{owner}/{repo}', or any of its variants with the forward slashes
    if not bool(re.match(r"^\/?[a-zA-Z0-9-_]+\/[a-zA-Z0-9-_]+\/?$", location)):
        return False

    owner, repo = get_owner_and_repo(location)

    # If the repo exists on GitHub and is public, the request will return a 200 status code
    if requests.get(f"https://api.github.com/repos/{owner}/{repo}").status_code != 200:
        return False
    return True


def get_type_of_location(location: typing.Union[str, os.PathLike]) -> str:
    """
    Get the type of location the repo is in.

    Parameters
    ----------
    location : str or os.PathLike
        The location to check.

    Returns
    -------
    str
        The type of location.
    """
    if is_local_git_repo(location):
        return "local"
    if is_github_repo(location):
        return "github"

    raise ValueError(f"{location} is not a valid location. Please check again.")
