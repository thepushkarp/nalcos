import os
import typing
from pathlib import Path
import numpy as np
from rich.console import Console
from appdirs import user_cache_dir
from git import Repo, Commit
from sentence_transformers import SentenceTransformer, util


def is_git_repo(directory: typing.Union[str, bytes, os.PathLike]) -> bool:
    try:
        repo = Repo(directory)
        return True
    except:
        return False


def get_commits(
    repo: Repo, branch: typing.Optional[str] = None
) -> typing.List[typing.Dict[str, str]]:
    if branch is None:
        branch = repo.head.ref.name

    commits = []

    try:
        for commit in repo.iter_commits(branch):
            commits.append(
                {
                    "author": str(commit.author),
                    "email": commit.author.email,
                    "message": commit.message.strip("\n"),
                    "id": commit.hexsha,
                    "commit_date": commit.committed_datetime,
                    "branch": branch,
                }
            )
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

    with console.status(f"[bold green]{status_message}...") as status:

        model = SentenceTransformer(model_name, cache_folder=cache_folder)

        commit_messages = [commit["message"] for commit in commits]

        embeddings = model.encode(commit_messages, convert_to_tensor=True)

    print("Indexed...")

    query_embedding = model.encode([query], convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings).squeeze()

    sorted_scores_indices = cosine_scores.argsort().tolist()[::-1][:n_matches]

    sorted_commits = [commits[i] for i in sorted_scores_indices]

    return sorted_commits
