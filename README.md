<h1 align="center">NaLCoS - NAtural Language COmmit Search</h1>

<p align="center">Search commit messages in your repository in <em>natural language</em>.</p>

<p align="center">
  <a href="https://github.com/thepushkarp/nalcos/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://github.com/thepushkarp/nalcos/stargazers"><img alt="Stargazers" src="https://img.shields.io/github/stars/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://github.com/thepushkarp/nalcos/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/thepushkarp/nalcos?style=for-the-badge"></a>
  <a href="https://github.com/thepushkarp/nalcos/releases"><img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/thepushkarp/nalcos?style=for-the-badge"></a>
</p>

---

NaLCoS (NAtural Language COmmit Search) is a command-line tool for searching commit messages in your repository in <em>natural language</em>.

The key features are:

- Search commit messages in both local and remote GitHub repositories.
- Search for commits in a specific branch.
- Restrict the number of commits to look back in history while searching.
- Increase the number of retrieved results.

![image](https://user-images.githubusercontent.com/42088801/133167351-d62accd0-34b3-4d22-a663-c7917556d16a.png)

Internally, NaLCoS uses [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) with pre-trained weights from [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1). I chose this particular model because it has a good [Performance vs Speed tradeoff](https://www.sbert.net/docs/pretrained_models.html). Since this model was designed for semantic search and has been pre-trained on 215M (question, answer) pairs from diverse sources, it is a good choice for tasks such as finding similarity between two sentences.

NaLCoS encodes the query string and all the commits into their corresponding vector embeddings and computes the cosine similarity between the query and all the commits. This is then used to rank the commits.

## Why did I build this?

Most of the times when I've used Machine Learning till now, has been in dedicated environments such as Google Colab or Kaggle. I had been learning Natural Language Processing for a while and wanted to use transformers to build something different that is not very resource (read GPU) intensive and can be used like an everyday tool.

Though many Transformer models are far from fitting this description, I found that distilled models are not as hungry as their older siblings are infamous for. Searching for Git commits using natural language was something on which I could not find any pre-existing tool and thus decided to give this a shot.

Though there are various improvements left, I'm happy with what this initially turned out to be. I'm eager to see what further enhancements can be made to this to make it more efficient and useful.

## Requirements

Tested on Python 3.8.11.

NaLCoS uses the following packages:

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for the Transformer model.
- [Git Python](https://github.com/gitpython-developers/GitPython) for local Git operations.
- [GitHub API](https://docs.github.com/en/rest) for interacting with GitHub repos.
- [Rich](https://github.com/willmcgugan/rich) for well-formatted CLI output.

## Installation

- Clone the repository:

```console
$ git clone https://github.com/thepushkarp/nalcos.git
```

This also downloads the model weights stored in the `nalcos/.cache` directory so you don't have to download them while running the model for the first time.

- Create a virtual environment ([click here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to read about activating virtualenv):

```
$ virtualenv venv
```

<details open>
<summary>Activate virtualenv (for Linux and MacOS):</summary>

```console
  $ source ./venv/bin/activate
```

</details>

<details open>
<summary>Activate virtualenv (for Windows):</summary>

```console
   $ cd venv/Scripts/
   $ activate
```

</details>

- Install the requirements:

```console
$ pip install -r requirements.txt
```

- Change directory to the `nalcos` directory:

```console
$ cd nalcos/
```

- Run NaLCoS:

```console
$ python nalcos.py [-g] [-n N_MATCHES] [-b BRANCH] [-l LOOK_PAST] "query string" "repository location"
```

## Usage

A detailed information about the usage of NaLCoS can be found below:

```
usage: nalcos [-h] [-g] [-n N_MATCHES] [-b BRANCH] [-l LOOK_PAST] [-v] query location

Search a commit in your git repository using natural language.

positional arguments:
  query                 The query to search for similar commit messages.
  location              The repository path to search in. If `-g` flag is not passed, searches locally in the path specified, else
                        takes in a remote GitHub repository name in the format '{owner}/{repo_name}'

optional arguments:
  -h, --help            show this help message and exit
  -g, --github          Flag to search on GitHub instead of searching in a local repository. Due to API limits currently this
                        allows for around 15 lookups per hour from your IP.
  -n N_MATCHES, --n-matches N_MATCHES
                        The number of matching results to return. Default 10.
  -b BRANCH, --branch BRANCH
                        The branch to search in. If not specified, the current branch will be used by default.
  -l LOOK_PAST, --look-past LOOK_PAST
                        Look back this many commits. Default 100.
  -v, --version         show program's version number and exit
```

### Examples

- Input:

```console
python nalcos.py "improve language" "github/docs" --github
```

- Output:

```
Found 100 commits.

                                        Commits related to "improve language" in "github/docs"
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ No. ┃ Commit ID ┃ Commit Message                                                        ┃ Commit Author      ┃ Commit Date          ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│  1. │ 51bfdbb95 │ Merge branch 'main' into fatenhealy-fix-supportedlanguage             │ Faten Healy        │ 2021-09-12T22:26:31Z │
│  2. │ a9c2c8eea │ fix deprecation label spelling (#21474)                               │ Rachael Sewell     │ 2021-09-13T18:12:03Z │
│  3. │ 94e3c092d │ English search sync (#21446)                                          │ Rachael Sewell     │ 2021-09-13T17:30:08Z │
│  4. │ b048e27e9 │ Merge pull request #9909 from github/fatenhealy-fix-supportedlanguage │ Ramya Parimi       │ 2021-09-12T22:35:19Z │
│  5. │ 73c2717f7 │ Fix typo                                                              │ Adrian Mato        │ 2021-09-13T06:35:27Z │
│  6. │ 86b571982 │ Export changes to a branch for codespaces (#21462)                    │ Matthew Isabel     │ 2021-09-13T14:55:50Z │
│  7. │ 969288662 │ Update diff limit to 500KB (#20616)                                   │ jjkennedy3         │ 2021-09-11T09:12:38Z │
│  8. │ f28ee46d4 │ Update OpenAPI Descriptions (#21447)                                  │ github-openapi-bot │ 2021-09-11T09:22:28Z │
│  9. │ 92af3a469 │ update search indexes                                                 │ GitHub Actions     │ 2021-09-12T09:50:46Z │
│ 10. │ e6018f2aa │ update search indexes                                                 │ GitHub Actions     │ 2021-09-11T02:05:19Z │
└─────┴───────────┴───────────────────────────────────────────────────────────────────────┴────────────────────┴──────────────────────┘

```


## Future plans

- [x] Documentation
- [x] Release first working version
- [ ] Publish to PyPi
- [ ] Add automated tests
- [ ] Add personal API token support to increase GitHub API rate limit
- [ ] Use a Python GitHub API wrapper [?]
- [ ] Look into ways to cache and store embeddings to reduce repeated computations [?]
- [ ] Try other models [?]

## Known issues

Not all retrieved results are always relevant. I could think of two primary reasons for this:

- The data the model was pre-trained on is not representative of how people write commit messages. Since commit messages usually contain technical jargon, merge commit messages, abbreviations and other non-common terms, the model (which has a limited vocabulary) is not able to generalize well to this data.
- Two commits may be related even when their commit messages may not be similar and similarly two commit messages maybe unrelated even when their commit messages are similar. We often need more metadata (such as lines changes, files changed) etc. to make the predictions more accurate.

## License

This project is licensed under the terms of the MIT license.

---

<p align="center">
  <a href="https://github.com/thepushkarp/nalcos" target="_blank" rel="noopener noreferrer">
    NaLCoS
  </a>
  made with ❤️ by
  <a href="https://github.com/thepushkarp">
    Pushkar  Patel
  </a>
</p>
