import random

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


def deep_model_preprocessing_csv(analogy_corpus,extract_all=False, ngram_abc="ngram_tokens.abc",ngram_d="ngram_tokens.d",base_address="./NMT_Analogies/data/"):
    print(len(analogy_corpus))
    ngram_trainabc= base_address + "ngram_train.csv"
    ngram_testabc= base_address + "ngram_test.csv"
    ngram_valabc= base_address + "ngram_val.csv"


    random.shuffle(analogy_corpus)
    splits = [100,120,140] #splits for train,val, test


    print("ngram train abc")
    with open(ngram_trainabc, 'w+') as f:
        for analogyList in analogy_corpus[: splits[0]]:
            sent = " ".join(token for token in analogyList[:-1]) + "," + analogyList[-1]
            f.write(sent + "\n")
        f.close()



    print("ngram val abc")
    with open(ngram_valabc, 'w+') as f:
        for analogyList in analogy_corpus[splits[0]: splits[1]]:
            sent = " ".join(token for token in analogyList[:-1]) + "," + analogyList[-1]
            f.write(sent + "\n")
        f.close()



    print("ngram test abc")
    with open(ngram_testabc, 'w+') as f:
        for analogyList in analogy_corpus[splits[1]: splits[2]]:
            sent = " ".join(token for token in analogyList[:-1]) + "," + analogyList[-1]
            f.write(sent + "\n")
        f.close()




def deep_model_preprocessing(analogy_corpus,extract_all=False, ngram_abc="ngram_tokens.abc",ngram_d="ngram_tokens.d",base_address="./NMT_Analogies/data/"):
    print(len(analogy_corpus))
    ngram_trainabc= base_address + "ngram_train.abc"
    ngram_traind= base_address + "ngram_train.d" 

    ngram_testabc= base_address + "ngram_test.abc"
    ngram_testd = base_address + "ngram_test.d"

    ngram_valabc= base_address + "ngram_val.abc"
    ngram_vald = base_address + "ngram_val.d"

    random.shuffle(analogy_corpus)
    splits = [100,120,140] #splits for train,val, test
    print(analogy_corpus[: splits[0]])
    if extract_all:
        print("Extract abc analogy")
        with open(ngram_abc, 'w+') as f:
            print(analogy_corpus[1])
            for analogyList in analogy_corpus:
                sent = " ".join(token for token in analogyList[:-1])
                f.write(sent + "\n")
            f.close()

        print("Extract d analogy")
        with open(ngram_d, 'w+') as f:
            print(analogy_corpus[1])
            for analogyList in analogy_corpus:
                f.write(analogyList[-1] + "\n")
            f.close()

    print("ngram train abc")
    with open(ngram_trainabc, 'w+') as f:
        for analogyList in analogy_corpus[: splits[0]]:
            sent = " ".join(token for token in analogyList[:-1])
            f.write(sent + "\n")
        f.close()
    print("ngram train d")
    with open(ngram_traind, 'w+') as f:
        for analogyList in analogy_corpus[: splits[0]]:
            f.write(analogyList[-1] + "\n")
        f.close()


    print("ngram val abc")
    with open(ngram_valabc, 'w+') as f:
        for analogyList in analogy_corpus[splits[0]: splits[1]]:
            sent = " ".join(token for token in analogyList[:-1])
            f.write(sent + "\n")
        f.close()
    print("ngram val d")
    with open(ngram_vald, 'w+') as f:
        for analogyList in analogy_corpus[splits[0]: splits[1]]:
            f.write(analogyList[-1] + "\n")
        f.close()


    print("ngram test abc")
    with open(ngram_testabc, 'w+') as f:
        for analogyList in analogy_corpus[splits[1]: splits[2]]:
            sent = " ".join(token for token in analogyList[:-1])
            f.write(sent + "\n")
        f.close()
    print("ngram test d")
    with open(ngram_testd, 'w+') as f:
        for analogyList in analogy_corpus[splits[1]: splits[2]]:
            f.write(analogyList[-1] + "\n")
        f.close()


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
    deep_model_preprocessing_csv(analogy_corpus, ngram_abc="ngram_tokens.abc",ngram_d="ngram_tokens.d" )
    # unigram_analogy = extract_unigram_analogies(analogy_corpus, "./data/unigram_tokens.txt")

if __name__ == '__main__':
    main()

