import numpy as np
import math

'''
This program main does 2 thing: generate a sentence base on the first word given and correct a sentence base on 
the observation given, Markov Model is using for generating a sentence and Hidden Markov model is used for correction

For generating a sentence: call function generate_sentence(vocab, unigram_probs, bigram_probs, trigram_probs, 
start) where vocab, unigram_probs, bigram_probs and trigram_probs are all given in main(), modify the starting words 
"start" to observe different result, be aware that the starting word in "start" has to be '<s>' 
(for testing, see line 222-223)

For correcting a sentence: call function correct_sentence(obs, state, start_pr, trans_pr, vocab) where parameters and 
some observations are provided in main() (you can also define your owned observation, but make sure it is the same 
format with the given examples, starting with '<s>') Testing for correcting each sentence might take a several minutes
(for testing, see line 227-242)

!!!!!!!!!
Be aware that:
 1. library Levenshtein is used for the calculation of distance between words
 2. viterbi_correction is implemented based on the pseudocode and code example provided on wikipedia
    https://en.wikipedia.org/wiki/Viterbi_algorithm
!!!!!!!!!
'''


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
    for key, value in dict.items():
        if value == tar_value:
            return key
    return None


def sample_word(vocab, unigram_probs, bigram_probs, trigram_probs, given_words=None):
    translate_word = []
    for word in given_words:
        translate_word.append(get_key_by_value(vocab, word))
    if len(translate_word) == 2:
        i, j = translate_word[0], translate_word[1]
        trigram_probs_row = trigram_probs.get(i, {}).get(j, 0)
        bigram_probs_row = bigram_probs.get(j, 0)
    else:
        i = translate_word[0]
        trigram_probs_row = {}
        bigram_probs_row = bigram_probs.get(i, 0)
    # Prioritize trigram, then bigram, and finally unigram if needed, back off
    if trigram_probs_row:
        next_token = max(trigram_probs_row.keys(), key=trigram_probs_row.get)
    elif bigram_probs_row:
        next_token = max(bigram_probs_row.keys(), key=bigram_probs_row.get)
    else:
        next_token = max(unigram_probs.keys(), key=unigram_probs.get)

    return next_token


def prior_sample(vocab, unigram_probs, bigram_probs, trigram_probs, start):
    sentence = []
    start = start.strip().split()
    sentence += start
    index = 0
    while sentence[-1] != "</s>" and index < 100:
        if len(sentence) - 1 >= 1:
            given_words = sentence[-2:]
        else:
            given_words = sentence[-1:]
        next_word = sample_word(vocab, unigram_probs, bigram_probs, trigram_probs, given_words)
        sentence.append(vocab[next_word])
        index += 1
    return sentence


def log_poisson_probability(u, v, lam):
    # Compute the Poinssson probability P(Et = u | Xt = v)
    # Emit P
    k = levenshtein(u, v)
    return k * np.log10(lam) - np.log10(math.factorial(k))


def viterbi_correction(obs, states, start_p, trans_p, vocab, lambd=0.01):
    # Implementation based on the pseudocode and code example on wikipedia
    # https://en.wikipedia.org/wiki/Viterbi_algorithm
    # lambd is by default set to 0.01
    V = [{}]
    for st in states:
        # since they are all log base, therefore multiplication become addition
        key = get_key_by_value(vocab, '<s>')
        log_emit_p = log_poisson_probability(obs[0], vocab[st], lambd)
        V[0][st] = {"prob": start_p[key] + log_emit_p, "prev": None}

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
                prev_trans_pr = trans_p.get(prev_st, {}).get(st, float('-inf'))
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


def levenshtein(s1, s2):
    # Dynamic Programming algorithm to compute the Levenshtein distance between two strings
    # Obtained from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def generate_sentence(vocab, unigram_probs, bigram_probs, trigram_probs, start):
    generated_sentence = prior_sample(vocab, unigram_probs, bigram_probs, trigram_probs, start)
    print(f"{'Generated Sentence: '} {' '.join(generated_sentence)}")


def correct_sentence(obs, state, start_pr, trans_pr, vocab):
    result = viterbi_correction(obs, state, start_pr, trans_pr, vocab)
    print(f"{'Original: '} {' '.join(obs)}")
    print(f"{'Corrected: '} {' '.join(result)}")


if __name__ == '__main__':
    # read in dictionary and probability files
    vocabularies = read_vocab_data()
    unigram = read_unigram_data()
    bigram = read_bigram_data()
    trigram = read_trigram_data()

    ''' generate a sentence giving a starting word '''
    starting_words = '<s> I hate'
    generate_sentence(vocabularies, unigram, bigram, trigram, starting_words)

    ''' Correct sentence given an observation (Testing may take a 1-2 min for each obs)'''
    # observations (in form of list, the starting char must be "<s>")
    obs1 = ["<s>", "I", "think", "hat", "twelve", "thousand", "pounds"]
    obs2 = ["<s>", "She", "haf", "heard", "them"]
    obs3 = ["<s>", "She", "was", "ulreedy", "quit", "live"]
    obs4 = ["<s>", "John", "knightly", "wasn't", "hard", "at", "word"]
    obs5 = ["<s>", "he", "said", "nit", "word", "by"]
    observations = [obs1, obs2, obs3, obs4, obs5]
    # parameters
    states = list(vocabularies.keys())
    start_p = unigram
    trans_p = bigram

    for observ in observations:
        correct_sentence(observ, states, start_p, trans_p, vocabularies)

    # if you do not want to test all of them, you can test them 1 by 1 as the following
    # correct_sentence(obs1, states, start_p, trans_p, vocabularies)
