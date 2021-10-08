import os
import pytest
from git import Repo, rmtree


@pytest.fixture(autouse=True, scope="session")
def create_local_repo():
    if os.path.exists("nalcos_pytest"):
        rmtree("nalcos_pytest")

    repo = Repo.clone_from(
        url="https://github.com/thepushkarp/nalcos.git",
        to_path="nalcos_pytest",
        branch="pytest",
    )

    yield

    repo.close()
    rmtree("nalcos_pytest")
