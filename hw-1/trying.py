import sys
import optparse
import sys

from collections import defaultdict


optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)


class Alignment:
    """
    A class to store the word alignment and its related information
    
    Attributes:
    f_sent (list): List of words in the foreign sentence
    e_sent (list): List of words in the target sentence
    f_index (dict): A dictionary where the keys are the words in f_sent and the values are the indices of those words
    e_index (dict): A dictionary where the keys are the words in e_sent and the values are the indices of those words
    """
    def __init__(self, f_sent, e_sent):
        """
        Initializes the Alignment class with the foreign and target sentences
        """
        self.f_sent = f_sent
        self.e_sent = e_sent
        self.f_index = defaultdict(list)
        self.e_index = defaultdict(list)
        # Indexing the words in f_sent
        for i, f_word in enumerate(f_sent):
            self.f_index[f_word].append(i)
        # Indexing the words in e_sent
        for i, e_word in enumerate(e_sent):
            self.e_index[e_word].append(i)


# Define the EM algorithm for IBM Model 1
def em_ibm1(sentences, iterations):
    """
    The EM algorithm for IBM Model 1

    Parameters:
    sentences (list): List of Alignment objects containing the foreign and target sentences
    iterations (int): Number of iterations for the EM algorithm

    Returns:
    dict: A dictionary where the keys are tuples of target word and foreign word and the values are the estimated translation probabilities t(e|f)
    """
    t = defaultdict(float)
    count = defaultdict(float)
    total = defaultdict(float)

    # Initialize t(e|f) uniformly
    for alignment in sentences:
        for f_word in alignment.f_sent:
            for e_word in alignment.e_sent:
                t[(e_word, f_word)] = 1.0 / len(alignment.e_sent)

    for i in range(iterations):
        # Initialize count(e|f) to
        count.clear()
        # Initialize total(f) to 0
        total.clear()

        # Compute normalization
        for alignment in sentences:
            for f_word in alignment.f_sent:
                s_total = 0
                for e_word in alignment.e_sent:
                    s_total += t[(e_word, f_word)]
                for e_word in alignment.e_sent:
                    c = t[(e_word, f_word)] / s_total
                    count[(e_word, f_word)] += c
                    total[f_word] += c

        # Re-estimate probabilities
        for e_word, f_word in count.keys():
            t[(e_word, f_word)] = count[(e_word, f_word)] / total[f_word]

    return t


def print_alignments(sentences, t):
    """
    Prints the alignments between the source and target sentences based on the translation probabilities.
    
    Parameters:
    sentences (list): A list of sentence objects, where each sentence object contains source and target sentences.
    t (dict): A dictionary representing the translation probabilities, with tuples of target and source words as keys and their probability as values.
    
    Returns:
    None
    """
    for alignment in sentences:
        for i, f_word in enumerate(alignment.f_sent):
            best_p = 0
            best_j = 0
            for j, e_word in enumerate(alignment.e_sent):
                p = t[(e_word, f_word)]
                if p > best_p:
                    best_p = p
                    best_j = j
            sys.stdout.write("%i-%i " % (i, best_j))
        sys.stdout.write("\n")

def main():
    """
    The main function that reads in the data, trains the IBM Model 1 using the Expectation Maximization (EM) algorithm,
    and prints the final alignments.
    
    Returns:
    None
    """
    # Read in the data
    f_sentences = [line.strip().split() for line in open(f_data, "r")][:opts.num_sents]
    e_sentences = [line.strip().split() for line in open(e_data, "r")][:opts.num_sents]
    sentences = [Alignment(f_sent, e_sent) for f_sent, e_sent in zip(f_sentences, e_sentences)]

    num_iterations = 5
    t = em_ibm1(sentences, num_iterations)
    print_alignments(sentences, t)
    
# Main function
if __name__ == "__main__":
    main()