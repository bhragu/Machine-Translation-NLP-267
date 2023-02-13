# NLP 267 Homework 1 README

## Prerequisite

### Installation

```bash
pip install collections tqdm
```

## Files

There are three python programs here (`-h` for usage):

- `./alignment.py` aligns words.

- `./check-alignments.py` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

- `./score-alignments.py` computes alignment error rate.

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

- `hansards.e` is the English side.

- `hansards.f` is the French side.

- `hansards.a` is the alignment of the first 37 sentences. The
  notation i-j means the word as position i of the French is
  aligned to the word at position j of the English. Notation
  i?j means they are probably aligned. Positions are 0-indexed.

## Usage

The commands work in a pipeline. For instance:

```bash
python alignment.py n -100000 -i 10 | python score-alignments.py
```

This command will run and evaluate IBM Model 1.

here,

- n = number of sentences to use for training and alignment
- i = number of iterations of em algorithm
