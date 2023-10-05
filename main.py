from Levenshtein import distance as dis
import numpy as np
import math

lam = 0.01


# Read data from unigram_counts
def read_unigram_data(file_path="./data/unigram_counts.txt"):
    unigram_model = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            index, prob = int(line[0]), float(line[1])
            unigram_model[index] = prob
    return unigram_model


# Read data from bigram_counts
def read_bigram_data(file_path="./data/bigram_counts.txt"):
    bigram_model = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            index_i, index_j, prob = int(line[0]), int(line[1]), float(line[2])
            if index_i not in bigram_model:
                bigram_model[index_i] = {}
            bigram_model[index_i][index_j] = prob
    return bigram_model


# Read data from trigram_counts
def read_trigram_data(file_path="./data/trigram_counts.txt"):
    trigram_model = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            index_i, index_j, index_k, prob = int(line[0]), int(line[1]), int(line[2]), float(line[3])
            if index_i not in trigram_model:
                trigram_model[index_i] = {}
            if index_j not in trigram_model[index_i]:
                trigram_model[index_i][index_j] = {}
            trigram_model[index_i][index_j][index_k] = prob
    return trigram_model


# Read data from vocab
def read_vocab_data(file_path="./data/vocab.txt"):
    vocab = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            index, word = int(line[0]), line[1]
            vocab[index] = word
    return vocab


def get_key_by_value(dict, tar_value):
    keys = []
    for key, value in dict.items():
        if value == tar_value:
            return key
    return None


def sample_word(vocab, unigram_probs, bigram_probs, trigram_probs, given_words=None):
    translate_word = []
    for word in given_words:
        translate_word.append(get_key_by_value(vocab, word))

    if len(given_words) == 2:
        i, j = translate_word[0], translate_word[1]
        trigram_probs_row = trigram_probs.get(i, {}).get(j, {})
        bigram_probs_row = bigram_probs.get(j, {})
    else:
        i = translate_word[0]
        trigram_probs_row = {}
        bigram_probs_row = bigram_probs.get(i, {})

    # Prioritize trigram, then bigram, and finally unigram if needed, back off
    if trigram_probs_row:
        next_token = max(trigram_probs_row.keys(), key=trigram_probs_row.get)
    elif bigram_probs_row:
        next_token = max(bigram_probs_row.keys(), key=bigram_probs_row.get)
    else:
        next_token = max(unigram_probs.keys(), key=unigram_probs.get)

    return next_token


def prior_sample(vocab, unigram_probs, bigram_probs, trigram_probs):
    sentence = ["<s>"]
    while sentence[-1] != "</s>":
        if len(sentence) > 1:
            given_words = sentence[-2:]
        else:
            given_words = sentence[-1:]
        next_word = sample_word(vocab, unigram_probs, bigram_probs, trigram_probs, given_words)
        print(next_word)
        sentence.append(vocab[next_word])
    return sentence


def poisson_probability(u, v, lam):
    # Compute the Poinssson probability P(Et = u | Xt = v)
    k = dis(u, v)
    return np.exp(-lam) * lam ** k / math.factorial(k)


def viterbi_correction(t, input_sentence, lam, vocab, bigram, corrected_list=None):
    words = input_sentence.split()
    vocabulary = vocab.values()
    if corrected_list is None:
        corrected_list = []

    if t == 0:
        starting_num = get_key_by_value(vocab, "<s>")
        temp = []
        for i, word in enumerate(vocabulary):
            word_num = get_key_by_value(vocab, word)
            x = bigram.get(starting_num).get(word_num)
            x = 0 if x is None else x
            y = poisson_probability("<s>", word, lam)
            temp.append(x + np.log10(y))
        max_i = np.argmax(temp)
        corrected_word = vocab[temp[max_i]]
        corrected_list.append(corrected_word)
        return corrected_list
    else:
        temp = []
        prev_word = viterbi_correction(t - 1, input_sentence, lam, vocab, bigram, corrected_list)[-1]
        starting_num = get_key_by_value(vocab, prev_word)
        for i, word in enumerate(vocabulary):
            word_num = get_key_by_value(vocab, word)
            x = bigram.get(starting_num).get(word_num)
            x = 0 if x is None else x
            y = poisson_probability(prev_word, word, lam)
            temp.append(x + np.log10(y))
        max_i = np.argmax(temp)
        corrected_word = vocab[temp[max_i]]
        corrected_list.append(corrected_word)
        return corrected_list


if __name__ == '__main__':
    vocab = read_vocab_data()
    unigram_probs = read_unigram_data()
    bigram_probs = read_bigram_data()
    trigram_probs = read_trigram_data()

    inputsentence = "she haf heard them "
    correct_sentence = viterbi_correction(len(inputsentence), inputsentence, lam, vocab, bigram_probs)
    print(correct_sentence)
