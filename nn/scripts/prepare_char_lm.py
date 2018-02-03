import argparse
import pickle
import numpy

from nn import utils

dataset = "ptb"


def make_vocab():
    char_vocab = utils.Vocabulary()
    with open('data/%s/%s.train.txt' % (dataset, dataset)) as f:
        chars = f.read()
    for i in range(len(chars)):
        ch = chars[i]
        char_vocab.add(ch)
        if i % 100000 == 0:
            print("\r%d of %d" % (i, len(chars)),end=' ')

    pickle.dump(char_vocab, open("data/char_vocab.pkl", "wb"))
    for i in range(len(char_vocab)):
        if i % 7 == 0:
            print()
        print(i, char_vocab.by_index(i), "  ",end=' ')


def convert(phase):
    vocab = pickle.load(open("data/char_vocab.pkl",'rb'))
    with open('data/%s/%s.%s.txt' % (dataset, dataset, phase)) as f:
        raw_chars = f.read()
    chars = numpy.zeros((len(raw_chars)))
    for i in range(len(raw_chars)):
        chars[i] = vocab.by_word(raw_chars[i])
    numpy.save("data/lm.%s.npy" % phase, chars.astype('int32'))
    print(phase, "done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='ptb')
    args = parser.parse_args()
    dataset = args.dataset
    print(dataset)

    make_vocab()

    convert("train")
    convert("valid")
    convert("test")

