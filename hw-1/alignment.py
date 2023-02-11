#!/usr/bin/env python
import optparse
import sys

import numpy as np

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

def train_ibm1(source_corpus, target_corpus, n_iters):
    sys.stdout.write("Training with IBM-1...\n")
    num_sentences = len(source_corpus)
    source_vocab = set()
    target_vocab = set()
    for source_sentence, target_sentence in zip(source_corpus, target_corpus):
        source_vocab.update(source_sentence)
        target_vocab.update(target_sentence)
    source_vocab = list(source_vocab)
    target_vocab = list(target_vocab)
    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)
    t = np.zeros((source_vocab_size, target_vocab_size)) + 1.0 / target_vocab_size
    
    for i in range(n_iters):
        # E-step: Compute the expected counts
        counts = np.zeros((source_vocab_size, target_vocab_size))
        total_counts = np.zeros(target_vocab_size)
        for source_sentence, target_sentence in zip(source_corpus, target_corpus):
            source_sentence_idx = [source_vocab.index(word) for word in source_sentence]
            target_sentence_idx = [target_vocab.index(word) for word in target_sentence]
            for source_word_idx in source_sentence_idx:
                normalization = np.sum([t[source_word_idx][j] for j in target_sentence_idx])
                for target_word_idx in target_sentence_idx:
                    counts[source_word_idx][target_word_idx] += t[source_word_idx][target_word_idx] / normalization
                    total_counts[target_word_idx] += t[source_word_idx][target_word_idx] / normalization
        # M-step: Update the alignment probabilities
        for j in range(target_vocab_size):
            for i in range(source_vocab_size):
                t[i][j] = counts[i][j] / total_counts[j]
    return t, source_vocab, target_vocab

def align_ibm1(source_sentence, target_sentence, t, source_vocab, target_vocab):
    source_sentence_idx = [source_vocab.index(word) for word in source_sentence]
    target_sentence_idx = [target_vocab.index(word) for word in target_sentence]
    alignment = []
    for source_word_idx in source_sentence_idx:
        max_prob = 0
        best_j = -1
        for target_word_idx in target_sentence_idx:
            if t[source_word_idx][target_word_idx] > max_prob:
                max_prob = t[source_word_idx][target_word_idx]
                best_j = target_word_idx
        alignment.append((source_word_idx, best_j))
    return alignment

def align_corpus(source_corpus, target_corpus, t, source_vocab, target_vocab):
    alignments = []
    for source_sentence, target_sentence in zip(source_corpus, target_corpus):
        alignments.append(align_ibm1(source_sentence, target_sentence, t, source_vocab, target_vocab))
    return alignments


def main():
    # Read in the data
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
    
    # source_corpus is a list of source sentences
    source_corpus = [pair[0] for pair in bitext]
    # target_corpus is a list of target sentences
    target_corpus = [pair[1] for pair in bitext]

    num_iterations = 5
    
    # Train IBM Model 1
    t, source_vocab, target_vocab = train_ibm1(source_corpus, target_corpus, n_iters=num_iterations)
    
    # Align the corpus using IBM Model 1
    alignments = align_corpus(source_corpus, target_corpus, t, source_vocab, target_vocab)
    
    for source_sentence, target_sentence, alignment in zip(source_corpus, target_corpus, alignments):
        print("Source Sentence:", " ".join(source_sentence))
        print("Target Sentence:", " ".join(target_sentence))
        print("Alignment:", alignment)
        print()

if __name__ == '__main__':
    main()
    