# Galytix Test

This repository contains the code for the Galytix test project.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Installation

To install the necessary dependencies, run the following command:

```bash
python -m build
```

## Usage
To start the application, use the following commands:

### Input files
Application expects the files to be located in the `resource` folder as:
 - *word2vec.csv*
 - *phrases.csv*

Alternatively you can use parameters to select different paths:
 - `--model_path` or `-m` for path to word2vec models
 - `--phrases_path` or `-p` for phrases

### Batch processing
Run the app with `-b` or `--batch` command
```bash
phrase-similarity -b
```

### Adhoc queries
Run the app with this command, you'll be able to enter your queries later.
```bash
phrase-similarity
```

### Help
Run it with `--help` argument