import argparse
from rich.console import Console
from rich.table import Table

from _version import __version__
from utils import get_type_of_location
from get_commits import get_local_commits, get_github_commits
from get_similar_commits import get_similar_commits


def main():
    """
    Main function
    """
    # Use argparse to define and get the arguments
    parser = argparse.ArgumentParser(
        prog="nalcos",
        description="Search a commit in your git repository using natural language.",
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
        help="Look back this many commits. Default 100.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-s",
        "--show-score",
        help="Shows the Cosine similarity score between the query and the retrieved commit messages. 1 is the best score and -1 is the worst.",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()

    console = Console()

    # Display a status bar while it retrieves the commits and computes similarity
    with console.status("[bold green]Retrieving the commits...") as status:
        # Get the location type from the location argument
        location_type = get_type_of_location(args.location)

        commits = None

        if args.github is True:
            # Checks if the location type is a GitHub repository
            if location_type != "github":
                raise ValueError(
                    "Repository not found. Please specify a GitHub repository in the format {owner}/{repo}."
                )
            commits = get_github_commits(args.location, args.look_past, args.branch)
        else:
            # Checks if the location type is a local repository
            if location_type != "local":
                raise ValueError(
                    "Repository not found. Please specify a local repository path."
                )
            commits = get_local_commits(args.location, args.look_past, args.branch)

        # If no commits found, raise error
        if commits in (None or []):
            raise ValueError("No commits found.")

        print(f"Found {len(commits)} commits.")
        print()
        # Get similar commits and display them as a clean table.
        similar_commits = get_similar_commits(args.query, commits, args.n_matches)
        table = Table(
            show_header=True,
            header_style="bold white",
            title=f'Commits related to "{args.query}" in "{args.location}"',
            title_style="bold white",
        )
        table.add_column("No.", justify="right")
        if args.show_score is True:
            table.add_column("Score")
        table.add_column("Commit ID")
        table.add_column("Commit Message")
        table.add_column("Commit Author")
        table.add_column("Commit Date")
        for i, commit in enumerate(similar_commits):
            row_to_add = [f"{i + 1}."]
            if args.show_score is True:
                row_to_add.append(commit["score"])
            row_to_add.extend(
                [
                    commit["id"][:9],
                    commit["message"].split("\n")[0],
                    commit["author"],
                    commit["commit_date"],
                ]
            )
            table.add_row(*row_to_add)
        console.print(table)


if __name__ == "__main__":
    main()
