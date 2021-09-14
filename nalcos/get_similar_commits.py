import os
import typing
from pathlib import Path
from appdirs import user_cache_dir
from sentence_transformers import SentenceTransformer, util

__all__ = ["get_similar_commits"]


def get_similar_commits(
    query: str, commits: typing.List[typing.Dict[str, str]], n_matches: int = 10
) -> typing.List[typing.Dict[str, str]]:
    """
    Get the n_matches most similar commits to the query commit.

    Arguments
    ---------
    query: str
        The query string to compare to.
    commits: typing.List[typing.Dict[str, str]]
        A list of commits to compare to the query.
    n_matches: int
        The number of matches to return.

    Returns
    -------
    typing.List[typing.Dict[str, str]]
        A list of the n_matches most similar commits to the query.
    """
    # The model to use for encoding the query and commit messages.
    # Pretrained models available at: https://www.sbert.net/docs/pretrained_models.html
    model_name = "multi-qa-MiniLM-L6-cos-v1"
    # The path to save the model to.
    cache_folder = user_cache_dir(
        os.path.join(Path(__file__).resolve().parent, "models")
    )

    # Load the model.
    model = SentenceTransformer(model_name, cache_folder=cache_folder)

    # Get all the commit messages.
    commit_messages = [commit["message"] for commit in commits]

    # Encode all the commit messages.
    commit_embeddings = model.encode(commit_messages, convert_to_tensor=True)

    # Encode the query.
    query_embedding = model.encode([query], convert_to_tensor=True)

    # Use cosine similarity to find the most similar commits.
    cosine_scores = util.pytorch_cos_sim(query_embedding, commit_embeddings).squeeze()

    # Get the indices of the most similar commits.
    sorted_scores_indices = cosine_scores.argsort().tolist()[::-1][:n_matches]

    # Get the n_matches most similar commits.
    sorted_commits = [commits[i] for i in sorted_scores_indices]

    return sorted_commits
