import os
import re
import typing
import requests
from pathlib import Path
from appdirs import user_cache_dir
from git import Repo
import torch
from sentence_transformers import SentenceTransformer

__all__ = [
    "get_owner_and_repo",
    "is_local_git_repo",
    "is_github_repo",
    "get_type_of_location",
    "get_model",
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

    status_code = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}"
    ).status_code
    # If the GitHub API has a rate limit, the status code will be 403
    if status_code == 403:
        raise ValueError(
            "GitHub API rate limit reached. Please wait a few minutes and try again."
        )
    # If the repo exists on GitHub and is public, the status code will be 200
    elif status_code == 200:
        return True
    return False


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


def get_model() -> SentenceTransformer:
    """
    Return the model used for the current sentence transformer.

    Returns
    -------
    SentenceTransformer
        The model.
    """
    # Pretrained models available at: https://www.sbert.net/docs/pretrained_models.html
    model_name = "multi-qa-MiniLM-L6-cos-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The path to save the model to.
    cache_folder = user_cache_dir(
        os.path.join(Path(__file__).resolve().parent, "models")
    )

    # Load the model.
    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_folder,
    )

    return model
