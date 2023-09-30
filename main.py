# Read date from unigram_date
def read_unigram_data(file_path="data/unigram_counts"):
    unigram_model = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            index, prob = int(line[0]), float(line[1])
            unigram_model[index] = prob
    return unigram_model


if __name__ == '__main__':
    a = read_unigram_data("data/unigram_counts")
    print(a)
