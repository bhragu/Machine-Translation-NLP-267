import sys
from collections import defaultdict
import optparse
import sys

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# Define a class to store the word alignment and its related information
class Alignment:
    def __init__(self, f_sent, e_sent):
        self.f_sent = f_sent
        self.e_sent = e_sent
        self.f_index = defaultdict(list)
        self.e_index = defaultdict(list)
        for i, f_word in enumerate(f_sent):
            self.f_index[f_word].append(i)
        for i, e_word in enumerate(e_sent):
            self.e_index[e_word].append(i)

# Define the EM algorithm for IBM Model 1
def em_ibm1(sentences, iterations):
    t = defaultdict(float)
    count = defaultdict(float)
    total = defaultdict(float)

    # Initialize t(e|f) uniformly
    for alignment in sentences:
        for f_word in alignment.f_sent:
            for e_word in alignment.e_sent:
                t[(e_word, f_word)] = 1.0 / len(alignment.e_sent)

    for i in range(iterations):
        # sys.stderr.write("Iteration %d\n" % (i + 1))

        # Initialize count(e|f) and total(f) to 0
        count.clear()
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

# Define a function to print the alignments
def print_alignments(sentences, t):
    for alignment in sentences:
        for i, f_word in enumerate(alignment.f_sent):
            best_p = 0
            best_j = 0
            for j, e_word in enumerate(alignment.e_sent):
                p = t[(e_word, f_word)]
                if p > best_p:
                    best_p = p
                    best_j = j
            sys.stdout.write("%d-%d " % (i, best_j))
        sys.stdout.write("\n")

def main():
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