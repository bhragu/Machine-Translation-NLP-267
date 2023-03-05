#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
import os
from io import open
import numpy as np


import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
from tqdm import tqdm


# Set up Logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


# Arguments Parser for Command Line Arguments
ap = argparse.ArgumentParser()
ap.add_argument('--seed', default=42, type=int, 
                help='random seed')
ap.add_argument('--hidden_size', default=256, type=int,
                help='hidden size of encoder/decoder, also word vector size')
ap.add_argument('--batch_size', default=8, type=int,
                help='batch size')
ap.add_argument('--num_epochs', default=5, type=int,
                help='num epochs')
ap.add_argument('--n_iters', default=8000, type=int,
                help='total number of examples to train on')
ap.add_argument('--teacher_forcing_rate', default=0.0, type=float,
                help='teacher forcing rate')
ap.add_argument('--print_every', default=5000, type=int,
                help='print loss info every this many training examples')
ap.add_argument('--dropout', default=0.2, type=float, 
                help='dropout rate')
ap.add_argument('--checkpoint_every', default=10000, type=int,
                help='write out checkpoint every this many training examples')
ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                help='initial learning rate')
ap.add_argument('--src_lang', default='fr',
                help='Source (input) language code, e.g. "fr"')
ap.add_argument('--tgt_lang', default='en',
                help='Source (input) language code, e.g. "en"')
ap.add_argument('--train_file', default='data/fren.train.bpe',
                help='training file. each line should have a source sentence,' +
                        'followed by "|||", followed by a target sentence')
ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                help='dev file. each line should have a source sentence,' +
                        'followed by "|||", followed by a target sentence')
ap.add_argument('--test_file', default='data/fren.test.bpe',
                help='test file. each line should have a source sentence,' +
                        'followed by "|||", followed by a target sentence' +
                        ' (for test, target is ignored)')
ap.add_argument('--out_file', default='out.txt',
                help='output file for test translations')
ap.add_argument('--load_checkpoint', nargs=1,
                help='checkpoint file to start from')

args = ap.parse_args()

# Set Device
def get_device() -> torch.device:
    """Returns the device to be used for model training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For new Mac M1 or M2 chips
    else:
        device = torch.device("cpu")
    logging.info(f"Using Device: {device}")
    return device

device = get_device()


def make_reproducible(seed: int=42) -> None:
    """Set seed to make the training reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15


class Vocab:
    """This class handles the mapping between the words and their indices
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, PAD_index: PAD_token}
        self.n_words = 3  # Count SOS, EOS and PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """Split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################


def tensor_from_sentence(vocab, sentence):
    """Creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """Creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    """the class for the encoder RNN"""
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def get_initial_hidden_state(self, bsz=None):
        D = 2 if self.bidirectional else 1
        if bsz is None:
            bsz = self.batch_size
        return torch.zeros(D, bsz, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout_p=0.1, max_length=MAX_LENGTH, bidirectional=False):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.effective_hidden_size = hidden_size*2 if bidirectional else self.hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(self.dropout_p)
        self.batch_size = batch_size

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.attn = nn.Linear(self.effective_hidden_size, self.effective_hidden_size)
        self.gru = nn.GRU(self.hidden_size*3 if self.bidirectional else self.hidden_size*2, 
                          hidden_size,
                          batch_first=True, 
                          bidirectional=self.bidirectional)
        self.out = nn.Linear(self.effective_hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        seq_len = encoder_outputs.shape[1]
        batch_size = input.shape[0]
        attn_energies = torch.zeros((batch_size, seq_len), device=device)
        for i in range(seq_len):
            energies = self.attn(encoder_outputs[:,i])
            attn_energies[:,i] =torch.bmm(
                                        hidden.view(batch_size, 1, self.effective_hidden_size), 
                                        energies.view(batch_size, self.effective_hidden_size, 1)
                                        ).squeeze(-1).squeeze(-1)
        attn_weights = F.softmax(attn_energies, dim=1)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)

        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, hidden)
        output = self.out(output)
        log_softmax = F.log_softmax(output, dim=-1)
        
        return log_softmax, hidden, attn_weights


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH, teacher_forcing_rate=0.0):
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    optimizer.zero_grad()
    
    # ENCODER
    encoder_hidden = encoder.get_initial_hidden_state()
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    # DECODER
    decoder_input = torch.zeros((encoder_outputs.shape[0], 1), device=device, dtype=torch.long)
    decoder_input[:,0] = SOS_index
    decoder_hidden = encoder_hidden.to(device)  # Use last hidden state from encoder to start decoder
    loss = 0

    for target_idx in range(target_tensor.size(1)):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output.squeeze(1), target_tensor[:,target_idx])
        teacher_force = random.uniform(0, 1)
        if teacher_force < teacher_forcing_rate:
            decoder_input = target_tensor[:,target_idx].unsqueeze(1)
        else:
            _, topi = decoder_output.data.topk(1)
            next_idxs = topi.squeeze(-1)
            decoder_input = next_idxs
    loss /= target_tensor.size(1)
    loss.backward()
    optimizer.step()

    return loss.item()


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state(bsz=1)
        encoder_outputs, encoder_hidden = encoder(input_tensor.T, encoder_hidden)

        decoder_input = torch.tensor([[SOS_index]], device=device, dtype=torch.long)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Adjust for max length vs length of sequence
            decoder_attentions[di,:decoder_attention.shape[-1]] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze(1).detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their translations
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, i=0):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    input_words = [SOS_token] + input_sentence.split()

    attn = attentions[:len(output_words), :len(input_words)].T
    
    heatmap = plt.imshow(attn, cmap="YlGnBu")
    # Create colorbar
    cbar = plt.colorbar(heatmap, label="Weights", ax=ax)

    ax.set_title("Translation Attentions")
    ax.set_xlabel("Target Sentence")
    ax.set_ylabel("Source Sentence")

    ax.set_xticks(np.arange(len(output_words)))
    ax.set_xticklabels(output_words, rotation=90)
    ax.set_yticks(np.arange(len(input_words)))
    ax.set_yticklabels(input_words)
    
    # Save Checkpoint
    path = "./plots"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, f"attentions-{i}.png")
    plt.savefig(filename)
    fig.clear()


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, i=0):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, i=i)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    # Set Seed for Reproducibility
    make_reproducible(args.seed)
    
    # Process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # Also set iteration 
    if args.load_checkpoint is not None:
        filename = os.path.join('./checkpoints', args.load_checkpoint[0])
        state = torch.load(filename)
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(input_size=src_vocab.n_words, 
                         hidden_size=args.hidden_size, 
                         batch_size=args.batch_size).to(device)
    decoder = AttnDecoderRNN(hidden_size=args.hidden_size, 
                             output_size=tgt_vocab.n_words, 
                             batch_size=args.batch_size, 
                             dropout_p=args.dropout).to(device)

    # Encoder/decoder weights are randomly initilized
    # If checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # Read in datafiles
    train_pairs = split_lines(args.train_file)[:args.n_iters]
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)
    
    # Log Stats about length of data
    logging.info(f"Number of training examples: {len(train_pairs)}")
    logging.info(f"Number of dev examples: {len(dev_pairs)}")
    logging.info(f"Number of test examples: {len(test_pairs)}")
    
    # Set up Optimization/Loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # Optimizer may have state
    # If checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])
    
    # Set up training data using vocab
    training_pairs = [tensors_from_pair(src_vocab, tgt_vocab, t) for t in train_pairs]
    
    class TrainDataset(Dataset):
        """Custom Dataset for loading train data."""
        def __init__(self, training_pairs) -> None:
            self.training_pairs = training_pairs
            self.input_pairs = [t[0] for t in self.training_pairs]
            self.target_pairs = [t[1] for t in self.training_pairs]
    
        def __len__(self) -> int:
            return len(self.training_pairs)
    
        def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
            return self.input_pairs[idx], self.target_pairs[idx]
        
    def custom_collate_fn(batch):
        """Custom collate function to pad sequences to the same length."""
        input_tensors, target_tensors = zip(*batch)
        input_tensors = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True, padding_value=PAD_index).squeeze(-1)
        target_tensors = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=PAD_index).squeeze(-1)
        return input_tensors, target_tensors
    
    # Batchify Training Data using DataLoader
    train_data = TrainDataset(training_pairs)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True, collate_fn=custom_collate_fn)

    # Train the model for a number of epochs
    for epoch in tqdm(range(args.num_epochs), desc=f"Training...", colour="red"):
        print_loss_total = 0 # Reset every epoch
        start = time.time()
        for _, (input_batch, target_batch) in tqdm(enumerate(train_loader)):
            loss = train(input_batch, 
                         target_batch, 
                         encoder,
                         decoder, 
                         optimizer, 
                         criterion, 
                         teacher_forcing_rate=args.teacher_forcing_rate)
            print_loss_total += loss

        # Log loss and time
        print_loss_avg = print_loss_total / len(train_loader)
        end = time.time()
        elapsed_time = end - start
        logging.info(f"Epoch {epoch+1:02} Loss: {print_loss_avg:.4}| Epoch {epoch+1:02} Time: {elapsed_time:.2f} sec")
        print("="*190)
        
        state = {'epoch_num': epoch+1,
                'enc_state': encoder.state_dict(),
                'dec_state': decoder.state_dict(),
                'opt_state': optimizer.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                }
        
        # Save Checkpoint
        path = "./checkpoints"
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, "state_%010d.pt" % epoch)
        torch.save(state, filename)
        logging.debug('Wrote checkpoint to %s', filename)
    
    # Translate from the dev set and compute BLEU score
    translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)
    references = [[clean(pair[1]).split(), ] for pair in dev_pairs]
    candidates = [clean(sent).split() for sent in translated_sentences]
    dev_bleu = corpus_bleu(references, candidates)
    logging.info('Dev BLEU score: %.2f', dev_bleu)    

    # Translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, i=0)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, i=1)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, i=2)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, i=3)


if __name__ == '__main__':
    main()
