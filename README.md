| **Authors**  | **Project** |  **Documentation** | **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:------------------:|:----------------:|:----------------:|:------------:|
| [**F. Colombo**](https://github.com/xover92) <br/> S&C26 student | **pr_PINN** | [![pr_PINN Docs CI](https://github.com/xover92/pr_PINN/actions/workflows/docs.yml/badge.svg)](https://github.com/xover92/pr_PINN/actions/workflows/docs.yml) | [![pr_PINN CI](https://github.com/xover92/pr_PINN/actions/workflows/python.yml/badge.svg)](https://github.com/xover92/pr_PINN/actions/workflows/python.yml) | [![Codacy Badge](https://app.codacy.com/project/badge/Grade/2fa4f86935e247069b6a95d5151fbc7f)](https://app.codacy.com/gh/xover92/pr_PINN/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) | **TODO** |

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/xover92/pr_PINN.svg?style=plastic)](https://github.com/xover92/pr_PINN/pulls)
[![GitHub issues](https://img.shields.io/github/issues/xover92/pr_PINN.svg?style=plastic)](https://github.com/xover92/pr_PINN/issues)

[![GitHub stars](https://img.shields.io/github/stars/xover92/pr_PINN.svg?label=Stars&style=social)](https://github.com/xover92/pr_PINN/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/xover92/pr_PINN.svg?label=Watch&style=social)](https://github.com/xover92/pr_PINN/watchers)

<a href="https://github.com/UniboDIFABiophysics">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# pr_PINN v0.0.1

## Project for the Pattern recognition and Software&Computing course (aa 2025-26)

This is a project developed for the Pattern recognition and Software&Computing courses of the Applied Physics curriculum.


* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Testing](#testing)
* [Table of contents](#table-of-contents)
* [Contribution](#contribution)
* [References](#references)
* [Authors](#authors)
* [License](#license)
* [Acknowledgments](#acknowledgments)
* [Citation](#citation)

## Overview

Write an overview about the context and/or project that you have developed.
In the documentation you can use also fancy layouts, tables, and references to the code.

| :triangular_flag_on_post: Note |
|:-------------------------------|
| This is an important note for your documentation! |

## Prerequisites

The complete list of requirements for the `pr_PINN` package is reported in the [requirements.txt](https://github.com/xover92/pr_PINN/blob/main/requirements.txt)

## Installation

Python version supported : ![Python version](https://img.shields.io/badge/python-3.5|3.6|3.7|3.8|3.9|3.10|3.11|3.12|3.13-blue.svg)

The `Python` installation for *developers* is executed using [`setup.py`](https://github.com/xover92/pr_PINN/blob/main/setup.py) script.

```mermaid
graph LR;
    A(Install<br>Requirements) -->|python -m pip install -r requirements.txt| B(Install<br>pr_PINN)
    B -->|python -m pip install .| C(Package<br>Install)
    B -->|python -m pip install --editable . --user| D(Development<br>Mode)
```

## Usage

You can use the `pr_PINN` library into your Python scripts or directly via command line.

### Command Line Interface

The `pr_PINN` package can be used directly via command line using the following syntax:

```bash
$ pr_PINN --help
usage: pr_PINN [-h] [--version] --input INPUT [--parallel {threads,processes}] [--num-workers NUM_WORKERS]

options:
  -h, --help            show this help message and exit
  --version, -v         Get the current version installed
  --input INPUT, -i INPUT
                        The input file from which to read the data. The file must be in CSV format with the column of
                        labels identified by the name "Y"; all the other columns will be interpreted as input
                        columns/features
  --parallel {threads,processes}, -p {threads,processes}
                        Parallelization scheme to use for the ML cross-validation
  --num-workers NUM_WORKERS, -n NUM_WORKERS
                        The number of worker threads/processes to use for parallel computation. Default is 4.
```

## Testing

**TODO**

## Table of contents

**TODO**

## Contribution

| :triangular_flag_on_post: Note |
|:-------------------------------|
| The following files are missing an they must be inserted/updated according to your needs/projects |

No contribution is allowed, since this is a project meant for university.

## References

<blockquote>1- Author et al, "Title", Journal, Year </blockquote>

## Authors

* <img src="https://avatars0.githubusercontent.com/u/24650975?s=400&v=4" width="25px"> [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="27px">](https://github.com/xover92) [<img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="25px">] **Francesco Colombo**

See also the list of [contributors](https://github.com/xover92/pr_PINN/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/xover92/pr_PINN.svg?style=plastic)](https://github.com/xover92/pr_PINN/graphs/contributors/) who participated in this project.

## License

The `pr_PINN` package is licensed under the GPLv3 [License](https://github.com/xover92/pr_PINN/blob/main/LICENSE).

## Acknowledgments

Thanks goes to all contributors of this project.

## Citation

If you have found `pr_PINN` helpful in your research, please consider citing the original repository

```BibTeX
@misc{pr_PINN,
  author = {Colombo, Francesco},
  title = {pr_PINN - Pattern Recognition exam: Physics Informed Neural Network},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/xover92/pr_PINN}}
}
```