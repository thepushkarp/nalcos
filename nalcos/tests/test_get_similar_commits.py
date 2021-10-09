from nalcos.utils import get_model
from nalcos.get_commits import get_local_commits
from nalcos.get_similar_commits import get_similar_commits

SORTED_COMMITS = [
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": ":sparkles: Adds a flag to show similarity scores of the result",
        "id": "3d30fd08dcabf1af0d3b88afb0d64f43e1c80674",
        "commit_date": "2021-09-18T08:07:16Z",
        "branch": "pytest",
        "score": "0.33",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": "Merge pull request #20 from thepushkarp/model_download\n\nImprove download prograss bar ddisplay",
        "id": "f4006bc05160eccac21a44fd14b2d5667efa3f0a",
        "commit_date": "2021-09-18T21:00:37Z",
        "branch": "pytest",
        "score": "0.15",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": "Merge pull request #19 from thepushkarp/visual-changes\n\nVisual improvements",
        "id": "b7a713ae76c4bbbd3c4b956b5fba91391b5a91be",
        "commit_date": "2021-09-18T19:43:12Z",
        "branch": "pytest",
        "score": "0.12",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": ":art: Improve download prograss bar display when loading for first time",
        "id": "72cd11b4ac1a755b3a705a6fd446980eaf61648f",
        "commit_date": "2021-09-18T20:59:33Z",
        "branch": "pytest",
        "score": "0.10",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": ":bookmark: Bump version to 0.2",
        "id": "22eb87fce284c10038ce223b589eedc520577b92",
        "commit_date": "2021-09-18T21:09:47Z",
        "branch": "pytest",
        "score": "0.07",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": "Merge pull request #21 from thepushkarp/dev\n\nVersion 0.2: Visual changes",
        "id": "2e410f45bad8a33cffa6b9ea0bf145006c580894",
        "commit_date": "2021-09-18T21:17:05Z",
        "branch": "pytest",
        "score": "0.05",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": ":sparkles:  Adds commit links for results from GitHub",
        "id": "c7e66e6cec0dcb74a85735a24b429e24c68cca98",
        "commit_date": "2021-09-18T19:40:16Z",
        "branch": "pytest",
        "score": "0.05",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": "Module error fixes",
        "id": "41fe13046baad8b4d5a526189eba9e3c1367037b",
        "commit_date": "2021-09-18T14:18:13Z",
        "branch": "pytest",
        "score": "0.02",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": "Update preview image",
        "id": "18d7ffb27c8263949c8cd86fd1e36a32ee622d42",
        "commit_date": "2021-09-18T21:13:26Z",
        "branch": "pytest",
        "score": "0.02",
    },
    {
        "author": "Pushkar Patel",
        "email": "42088801+thepushkarp@users.noreply.github.com",
        "message": ":sparkles: Adds an flag to display the entire commit message",
        "id": "1c3ebff5bb4bf4fa9272cc469d7e5f4560e4309e",
        "commit_date": "2021-09-18T08:17:53Z",
        "branch": "pytest",
        "score": "-0.00",
    },
]


def test_get_similar_commits():
    model = get_model()
    commits = get_local_commits("nalcos_pytest", 10, "pytest")

    sorted_commits = get_similar_commits(model, "Improve score", commits)

    assert sorted_commits == SORTED_COMMITS
