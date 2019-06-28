from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def preprocess(data):
    
    # build sentences
    agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                s["tag"].values.tolist())]

    grouped = data.groupby("sentence_idx").apply(agg_func)
    sentences = [s for s in grouped]
    sentences_list = [" ".join([s[0] for s in sent]) for sent in sentences]
    print(sentences_list[0])
    print(sentences[0])

    # calculate maximum sentence length
    maxlen = max([len(s) for s in sentences])
    print ('Maximum sequence length:', maxlen)

    # calculate number of words
    words = list(set(data["word"].values))
    #words.append("ENDPAD")
    n_words = len(words)
    print ('Number of words:', n_words)

    # calculate number of tags
    tags = list(set(data["tag"].values))
    n_tags = len(tags)
    print ('Number of tags:', n_tags)
    print ('Type of tags:', tags)

    # tokenize features
    tokenizer = Tokenizer(lower=False, filters=[], oov_token=None)
    tokenizer.fit_on_texts(sentences_list)
    number_words = len(tokenizer.word_index)
    
    sequences = tokenizer.texts_to_sequences(sentences_list)
    X = pad_sequences(maxlen=140, sequences=sequences, padding="post",value=number_words - 1)

    # one hot encoding labels
    tag2idx = {t: i for i, t in enumerate(tags)}
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    
    return n_words, n_tags, X, y