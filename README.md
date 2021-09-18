<h1 align="center">NaLCoS - NAtural Language COmmit Search</h1>

<p align="center">Search commit messages in your repository in <em>natural language</em>.</p>

<p align="center">
  <a href="https://github.com/thepushkarp/nalcos/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://github.com/thepushkarp/nalcos/stargazers"><img alt="Stargazers" src="https://img.shields.io/github/stars/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://github.com/thepushkarp/nalcos/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge"></a>
  <br>
   <a href="https://github.com/thepushkarp/nalcos/releases"><img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://pypi.org/project/nalcos/"><img alt="PyPi" src="https://img.shields.io/pypi/v/nalcos?style=for-the-badge"></a>
  <br>
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
  <a href="#contributors"><img alt="All contributors" src="https://img.shields.io/badge/all_contributors-1-orange.svg?style=for-the-badge"></a>
  <!-- ALL-CONTRIBUTORS-BADGE:END -->
</p>

---

NaLCoS (NAtural Language COmmit Search) is a command-line tool for searching commit messages in your repository in <em>natural language</em>.

The key features are:

- Search commit messages in both local and remote GitHub repositories.
- Search for commits in a specific branch.
- Look back any number of commits in history.
- Open the commit link directly in your browser if retrieved from GitHub.

![image](https://user-images.githubusercontent.com/42088801/133908654-9cec1f5c-6e9f-4eba-9254-8383e05c3824.png)

Internally, NaLCoS uses [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) with pre-trained weights from [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1). I chose this particular model because it has a good [Performance vs Speed tradeoff](https://www.sbert.net/docs/pretrained_models.html). Since this model was designed for semantic search and has been pre-trained on 215M (question, answer) pairs from diverse sources, it is a good choice for tasks such as finding similarity between two sentences.

NaLCoS encodes the query string and all the commits into their corresponding vector embeddings and computes the cosine similarity between the query and all the commits. This is then used to rank the commits.

## Why did I build this?

Most of the times when I've used Machine Learning till now, has been in dedicated environments such as Google Colab or Kaggle. I had been learning Natural Language Processing for a while and wanted to use transformers to build something different that is not very resource (read GPU) intensive and can be used like an everyday tool.

Though many Transformer models are far from fitting this description, I found that distilled models are not as hungry as their older siblings are infamous for. Searching for Git commits using natural language was something on which I could not find any pre-existing tool and thus decided to give this a shot.

Though there are various improvements left, I'm happy with what this initially turned out to be. I'm eager to see what further enhancements can be made to this to make it more efficient and useful.

## Requirements

NaLCoS uses the following packages:

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for the Transformer model.
- [Git Python](https://github.com/gitpython-developers/GitPython) for local Git operations.
- [GitHub API](https://docs.github.com/en/rest) for interacting with GitHub repos.
- [Rich](https://github.com/willmcgugan/rich) for well-formatted CLI output.

## Installation

### Installing with `pip` (Recommended)

Install with `pip` or your favourite PyPi manager:

```console
$ pip install nalcos
```

Run NaLCoS on a repository of your choice. For example:

```console
$ nalcos "handle nan issues" "numpy/numpy" --github
```

To see all available options, run with the `--help` flag:

```console
$ nalcos --help
```

**Note**: When you run the `nalcos` command for the first time, it will, download the model which would be cached and used the next time you run NaLCoS.

### Installing bleeding edge from the GitHub repository

- Clone the repository:

```console
$ git clone https://github.com/thepushkarp/nalcos.git
```

This also downloads the model weights stored in the `nalcos/models` directory so you don't have to download them while running the model for the first time.

- cd into the `nalcos` directory:

```console
$ cd nalcos
```

- Create a virtual environment ([click here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to read about activating virtualenv):

```
$ virtualenv venv
```

- Activate virtualenv (for Linux and MacOS):

```console
$ source ./venv/bin/activate
```

- Activate virtualenv (for Windows):

```console
$ cd venv/Scripts/
$ activate
```

- Install the requirements and the module:

```console
$ pip install -r requirements.txt
$ pip install -e .
```

Run NaLCoS on a repository of your choice. For example:

```console
$ nalcos "handle nan issues" "numpy/numpy" --github
```

To see all available options, run with the `--help` flag:

```console
$ nalcos --help
```

## Usage

A detailed information about the usage of NaLCoS can be found below:

```
usage: nalcos [-h] [-g] [-n N_MATCHES] [-b BRANCH] [-l LOOK_PAST] [-s] [-v] [--version] query location

Search a commit in your git repository using natural language.

positional arguments:
  query                 The query to search for similar commit messages.
  location              The repository path to search in. If '-g' or '--github' flag is not passed, searches
                        locally in the path specified, else takes in a remote GitHub repository name in the
                        format '{owner}/{repo_name}'

optional arguments:
  -h, --help            show this help message and exit
  -g, --github          Search on GitHub instead of searching in a local repository. Due to API limits
                        currently this allows for around 15 lookups per hour from your IP.
  -n N_MATCHES, --n-matches N_MATCHES
                        The number of matching results to return. Default 10.
  -b BRANCH, --branch BRANCH
                        The branch to search in. If not specified, the current branch will be used by default.
  -l LOOK_PAST, --look-past LOOK_PAST
                        Look back this many commits. Default 100.
  -s, --show-score      Shows the Cosine similarity score between the query and the retrieved commit messages.
                        1 is the best match and -1 is the worst.
  -v, --verbose         Show the entire commit message and not just the commit title.
  --version             show program's version number and exit
```

### Examples

- Input:

```console
$ nalcos "handle nan issues" "numpy/numpy" --github
```

- Output:

```
Found 100 commits.

                                                  Commits related to "handle nan issues" in "numpy/numpy"
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ No. ┃ Commit ID ┃ Commit Message                                                                            ┃ Commit Author       ┃ Commit Date          ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│  1. │ b6d7c4680 │ BUG: Fixed an issue wherein certain `nan<x>` functions could fail for object arrays       │ Bas van Beek        │ 2021-09-03T13:41:54Z │
│  2. │ e4f85b08c │ Merge pull request #19863 from BvB93/nanquantile                                          │ Charles Harris      │ 2021-09-13T23:21:51Z │
│  3. │ ecba7133f │ MAINT: Let `_remove_nan_1d` attempt to identify nan-containing object arrays              │ Bas van Beek        │ 2021-09-05T21:46:34Z │
│  4. │ 95e5d5abb │ BUG: Fixed an issue wherein `nanpercentile` and `nanquantile` would ignore the dtype for  │ Bas van Beek        │ 2021-09-11T11:54:56Z │
│     │           │ all-nan arrays                                                                            │                     │                      │
│  5. │ b3a66e88b │ Merge pull request #19821 from BvB93/nanfunctions                                         │ Charles Harris      │ 2021-09-05T23:32:30Z │
│  6. │ dc7dafe70 │ Merge pull request #19869 from mhvk/median_scalar_nan                                     │ Charles Harris      │ 2021-09-14T21:09:26Z │
│  7. │ 9ef778330 │ TST: Add more tests for `nanmedian`, `nanquantile` and `nanpercentile`                    │ Bas van Beek        │ 2021-09-03T15:01:57Z │
│  8. │ 6ba48721e │ BUG: ensure np.median does not drop subclass for NaN result.                              │ Marten van Kerkwijk │ 2021-09-13T19:50:54Z │
│  9. │ e62aa4968 │ Merge pull request #19854 from BvB93/nanfunctions                                         │ Charles Harris      │ 2021-09-09T15:14:09Z │
│ 10. │ 268e8e885 │ TST: Make nanfunc test ignore overflow instead of xfailing test                           │ Sebastian Berg      │ 2021-09-07T22:55:41Z │
└─────┴───────────┴───────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────┴──────────────────────┘
```

## Future plans

Please visit the [NaLCoS To Do Project Board](https://github.com/thepushkarp/nalcos/projects/1) to see current status and future plans.

## Known issues

Not all retrieved results are always relevant. I could think of two primary reasons for this:

- The data the model was pre-trained on is not representative of how people write commit messages. Since commit messages usually contain technical jargon, merge commit messages, abbreviations and other non-common terms, the model (which has a limited vocabulary) is not able to generalize well to this data.
- Two commits may be related even when their commit messages may not be similar and similarly two commit messages maybe unrelated even when their commit messages are similar. We often need more metadata (such as lines changes, files changed) etc. to make the predictions more accurate.

## Contributing

Any suggestions, improvements or bug reports are welcome.

- If you want to discuss any aspect of the project, please use the [Discussions Tab](https://github.com/thepushkarp/nalcos/discussions).
- You can submit your idea by [opening an issue](https://github.com/thepushkarp/nalcos/issues/new/choose) or [creating a Pull Request](https://github.com/thepushkarp/nalcos/pulls).
- If you'd like to improve the code, make sure you stick to the existing code style and naming conventions.

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://thepushkarp.com/"><img src="https://avatars.githubusercontent.com/u/42088801?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pushkar Patel</b></sub></a><br /><a href="https://github.com/thepushkarp/nalcos/commits?author=thepushkarp" title="Code">💻</a> <a href="https://github.com/thepushkarp/nalcos/commits?author=thepushkarp" title="Documentation">📖</a> <a href="#maintenance-thepushkarp" title="Maintenance">🚧</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

This project is licensed under the terms of the MIT license.
