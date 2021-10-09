import numpy as np
from numpy.testing import assert_array_almost_equal_nulp
import pytest
from nalcos.utils import (
    get_owner_and_repo,
    is_local_git_repo,
    is_github_repo,
    get_type_of_location,
)


def test_get_owner_and_repo():
    assert get_owner_and_repo("thepushkarp/nalcos") == ("thepushkarp", "nalcos")
    assert get_owner_and_repo("thepushkarp/nalcos/") == ("thepushkarp", "nalcos")
    assert get_owner_and_repo("/thepushkarp/nalcos") == ("thepushkarp", "nalcos")
    assert get_owner_and_repo("/thepushkarp/nalcos/") == ("thepushkarp", "nalcos")


def test_is_local_git_repo():
    assert is_local_git_repo("nalcos_pytest")
    assert not is_local_git_repo("not_nalcos")


def test_is_github_repo():
    assert is_github_repo("thepushkarp/nalcos")
    assert is_github_repo("thepushkarp/nalcos/")
    assert is_github_repo("/thepushkarp/nalcos")
    assert is_github_repo("/thepushkarp/nalcos/")
    assert not is_github_repo("thepushkarp/not_nalcos")
    assert not is_github_repo("thepushkarp/not_nalcos/test")
    assert not is_github_repo("nalcos_pytest")
    assert not is_github_repo("thepushkarp/private")


def test_get_type_of_location():
    assert get_type_of_location("nalcos_pytest") == "local"
    assert get_type_of_location("thepushkarp/nalcos") == "github"
    with pytest.raises(ValueError):
        get_type_of_location("thepushkarp/not_nalcos")
    with pytest.raises(ValueError):
        get_type_of_location("not_nalcos")
