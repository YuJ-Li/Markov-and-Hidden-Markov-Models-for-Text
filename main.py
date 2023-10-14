import Levenshtein
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
        sentence.append(vocab[next_word])
    return sentence


def log_poisson_probability(u, v, lam):
    # Compute the Poinssson probability P(Et = u | Xt = v)
    # Emit P
    k = Levenshtein.distance(u, v)
    return k * np.log10(lam) - np.log10(math.factorial(k))


def viterbi_correction(obs, states, start_p, trans_p, vocab, lambd = 0.01):
    # Considering the code example on wikipedia https://en.wikipedia.org/wiki/Viterbi_algorithm
    V = [{}]
    for st in states:
        # since they are all log base, therefore multiplication become addition
        log_emit_p = log_poisson_probability(obs[0], vocab[st], lambd)
        V[0][st] = {"prob": start_p[st] + log_emit_p, "prev": None}

    # Run Viterbi when t>0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            # Find the maximum probability for the current state at time t
            log_emit_p = log_poisson_probability(obs[t], vocab[st], lambd)
            trans_pr = trans_p.get(states[0], {}).get(st, float('-inf'))
            max_tr_prob = V[t - 1][states[0]]["prob"] + trans_pr + log_emit_p
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                prev_trans_pr = trans_p.get(prev_st, {}).get(st,float('-inf'))
                tr_prob = V[t - 1][prev_st]["prob"] + prev_trans_pr + log_emit_p
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
            max_prob = max_tr_prob
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
    # Backtrack to find the most likely sequence
    opt = []
    max_prob = float('-inf')  # Updated to negative infinity
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    sentence = []
    for word in opt:
        sentence.append(vocab[word])
    return sentence


if __name__ == '__main__':
    vocab = read_vocab_data()
    unigram_probs = read_unigram_data()
    bigram_probs = read_bigram_data()
    trigram_probs = read_trigram_data()

    # Generate sentence
    generated_sentence = prior_sample(vocab, unigram_probs, bigram_probs, trigram_probs)
    print(" ".join(generated_sentence))

    # Correct sentence
    obs1 = ["<s>", "I", "think", "hat", "twelve", "thousand", "pounds"]
    obs2 = ["<s>", "She", "haf", "heard", "them"]
    obs3 = ["<s>", "She", "was", "ulreedy", "quit", "live"]
    obs4 = ["<s>", "John", "knightly", "wasn't", "hard", "at", "word"]
    obs5 = ["<s>", "he", "said", "nit", "word", "by"]
    states = list(vocab.keys())
    start_p = unigram_probs
    trans_p = bigram_probs
    result = viterbi_correction(obs5, states, start_p, trans_p, vocab)
    print("Most likely sequence:", result)
