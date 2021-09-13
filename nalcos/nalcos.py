import argparse
from rich.console import Console
from rich.table import Table

from utils import get_type_of_location
from get_commits import get_local_commits, get_github_commits
from get_similar_commits import get_similar_commits


def main():
    # Use argparse to define and get the arguments
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

    console = Console()

    # Display a status bar while it retrieves the commits and computes similarity
    with console.status("[bold green]Retreiving your commits...") as status:
        # Get the location type from the location argument
        location_type = get_type_of_location(args.location)

        commits = None

        if args.github == True:
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
        if commits == None or commits == []:
            raise ValueError("No commits found.")
        else:
            print(f"Found {len(commits)} commits.")
            print()

            # Get similar commits and display them as a clean table.
            similar_commits = get_similar_commits(args.query, commits, args.n_matches)

            table = Table(show_header=True, header_style="bold white")
            table.add_column("Commit ID", style="dim")
            table.add_column("Commit Message")
            table.add_column("Commit Author")
            table.add_column("Commit Date")

            for commit in similar_commits:
                table.add_row(
                    commit["id"][:9],
                    commit["message"],
                    commit["author"],
                    commit["commit_date"],
                )

            console.print(table)


if __name__ == "__main__":
    main()