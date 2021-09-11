import os
import typing
import argparse
import requests
from pathlib import Path
import numpy as np
from rich.console import Console
from appdirs import user_cache_dir
from git import Repo, Commit
import torch
from sentence_transformers import SentenceTransformer, util
from pprint import pprint
from datetime import timezone


def get_owner_and_repo(location: str) -> typing.Tuple[str, str]:
    owner, repo = location.split("/")[:2]
    return (owner, repo)


def is_local_git_repo(location: typing.Union[str, bytes, os.PathLike]) -> bool:
    try:
        _ = Repo(location)
        return True
    except:
        return False


def is_github_repo(location: str) -> bool:
    owner, repo = get_owner_and_repo(location)
    if requests.get(f"https://api.github.com/repos/{owner}/{repo}").status_code != 200:
        return False
    return True


def get_type_of_location(location: typing.Union[str, bytes, os.PathLike]) -> str:
    if is_local_git_repo(location):
        return "local"
    elif is_github_repo(location):
        return "github"
    else:
        raise ValueError(f"{location} is not a valid location. Please check again.")


def get_local_commits(
    location: typing.Union[str, bytes, os.PathLike],
    max_count: int = 1000,
    branch: typing.Optional[str] = None,
) -> typing.List[typing.Dict[str, str]]:
    repo = Repo(location)
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
                    "commit_date": commit.committed_datetime.astimezone(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "branch": branch,
                }
            )
    except:
        pass

    return commits


def get_github_commits(
    location: str, max_count: int = 1000, branch: typing.Optional[str] = None
) -> typing.List[typing.Dict[str, str]]:
    owner, repo = get_owner_and_repo(location)

    if branch is None:
        response = requests.get(f"https://api.github.com/repos/{owner}/{repo}").json()
        branch = response["default_branch"]

    all_branches_response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/branches"
    ).json()
    all_branches_names = [
        remote_branch["name"] for remote_branch in all_branches_response
    ]

    commits = []

    if branch in all_branches_names:
        branch_hash = all_branches_response[all_branches_names.index(branch)]["commit"][
            "sha"
        ]

        num_pages = int(np.ceil(max_count / 100))
        i = 1
        run_get_commits_loop = True

        try:
            while run_get_commits_loop:
                commits_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch_hash}&per_page=100?page={i}"
                ).json()
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
            commits = commits[:max_count]
        except:
            pass

    return commits


def get_similar_commits(
    query: str, commits: typing.List[typing.Dict[str, str]], n_matches: int = 10
) -> typing.List[typing.Dict[str, str]]:
    model_name = "all-MiniLM-L6-v2"
    cache_folder = user_cache_dir(
        os.path.join(Path(__file__).resolve().parent, ".cache")
    )

    status_message = np.random.choice(
        [
            "Performing matrix magic",
            "Downloading some more RAM. Hold tight",
            "Alexa, compute embeddings",
            "Looking through some interesting messages",
            "Calling in the power of the BERT",
        ]
    )

    console = Console()

    with console.status(f"[bold white]{status_message}...") as status:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(
            model_name, cache_folder=cache_folder, device=device
        )

        commit_messages = [commit["message"] for commit in commits]

        embeddings = model.encode(commit_messages, convert_to_tensor=True)

    print("Indexed...")

    query_embedding = model.encode([query], convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings).squeeze()

    sorted_scores_indices = cosine_scores.argsort().tolist()[::-1][:n_matches]

    sorted_commits = [commits[i] for i in sorted_scores_indices]

    return sorted_commits


def main():
    parser = argparse.ArgumentParser(
        description="Search a commit in your git repository using natural language."
    )
    parser.add_argument(
        "query", help="The query to search for similar commit messages.", type=str
    )
    parser.add_argument(
        "location",
        help="The repository path to search in. If `-g` flag is not passed, searches locally in the path specified, else takes in a remote GitHub repository name in the format '{owner}/{repo_name}'",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--github",
        help="Flag to search on GitHub instead of searching in a local repository. Due to API limits currently this allows for around 15 lookups per hour from your IP.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--n-matches",
        help="The number of matching results to return. Default 10.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-b",
        "--branch",
        help="The branch to search in. If not specified, the current branch will be used by default.",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--look-past",
        help="Look back this many commits. Default 1000.",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    print(args)
    print()

    # Sanity checks
    location_type = get_type_of_location(args.location)

    if args.github == True:
        if location_type != "github":
            raise ValueError(
                "Repository not found. Please specify a GitHub repository in the format {owner}/{repo}."
            )
        commits = get_github_commits(args.location, args.look_past, args.branch)
        pprint(commits)
        pprint(len(commits))
    else:
        if location_type != "local":
            raise ValueError(
                "Repository not found. Please specify a local repository path."
            )
        commits = get_local_commits(args.location, args.look_past, args.branch)
        pprint(commits)
        pprint(len(commits))


if __name__ == "__main__":
    main()
