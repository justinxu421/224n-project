# Read the analogies from the text file
def read_analogies(analogy_text, to_lower = True):
    sents = []
    f_new = open('./all-data.txt', 'w+')
    print("Reading in analogy")
    with open(analogy_text) as f:
        for sent in f:
            cur_sent = sent[:]
            if (to_lower):
                cur_sent = cur_sent.lower()
            cur_sent = sent.strip().split(' | ')[1:]
            if cur_sent not in sents:
                f_new.write(sent)
                sents.append(cur_sent)
    return sents

# Extract unigram analogy from corpus extracted (list of analogy)
def extract_unigram_analogies(analogy_corpus, unigram_file="unigram_tokens.txt"):
    unigram_analogy = []
    print("Extracting unigram analogy")
    with open(unigram_file, 'w+') as f:
        print(analogy_corpus[1])
        for analogyList in analogy_corpus:
            is_unigram = True
            for word in analogyList:
                if ' ' in word:
                    is_unigram = False
                    break
            if is_unigram:
                unigram_analogy.append(analogyList)
                sent = " ".join(token for token in analogyList)
                f.write(sent + "\n")
        f.close()
    return unigram_analogy


def main():
    analogy_corpus = read_analogies('./data/analogy_data/all-analogy-unique.txt');
    unigram_analogy = extract_unigram_analogies(analogy_corpus, "./data/unigram_tokens.txt")

if __name__ == '__main__':
    main()

