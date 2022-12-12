import math,re

eps = 0.257
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class UnigramLanguageModel:
    def __init__(self, file, smoothing=False):
        with open(file, "r") as f:
            self.sentences=[re.split("\s+", line.rstrip('\n')) for line in f]
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in self.sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing

    def in_vocab(self, word):
        if word not in self.unigram_frequencies:
                self.unigram_frequencies['UNK']=self.unigram_frequencies.get('UNK', 0) + 1
        return word in self.unigram_frequencies

    def calculate_unigram_probability(self, word):
            if word not in self.unigram_frequencies:
                self.unigram_frequencies['UNK']=self.unigram_frequencies.get('UNK', 0) + 1
            word_probability_numerator = self.unigram_frequencies.get(word, 0)
            word_probability_denominator = self.corpus_length
            if self.smoothing:
                word_probability_numerator += 1
                # add one more to total number of seen unique words for UNK - unseen events
                word_probability_denominator += self.unique_words + 1
            return (math.log(word_probability_numerator) - math.log(word_probability_denominator))

    def calculate_sentence_probability(self, sentence):
        sentence_probability_log_sum = 0
        for word in sentence:
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability = self.calculate_unigram_probability(word)
                sentence_probability_log_sum += word_probability
        return sentence_probability_log_sum                

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.remove(SENTENCE_START)
        full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

    def log_prob(self, word):
        if word in self.unigram_frequencies:
            return math.log(self.unigram_frequencies[word]) - math.log(self.corpus_length)
        else:
            return float("-inf")

    def check_probs(self):

        for w in self.unigram_frequencies:
            assert 0 - eps < math.exp(self.log_prob(w)) < 1 + eps
        assert 1 - eps < \
            sum([math.exp(self.log_prob(w)) for w in self.unigram_frequencies]) < \
            1 + eps

class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, file, smoothing=False):
        UnigramLanguageModel.__init__(self, file, smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in self.sentences:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),
                                                                                                 0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_bigrams.add((previous_word, word))
                previous_word = word
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)
        if self.smoothing:
            bigram_word_probability_numerator += 1
            bigram_word_probability_denominator += self.unique__bigram_words + 1
            
        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else(math.log(bigram_word_probability_numerator) - math.log(bigram_word_probability_denominator))

    def calculate_bigram_word_probability(self, sentence,index):
        if sentence[index] not in self.unigram_frequencies:
                self.unigram_frequencies['UNK']=self.unigram_frequencies.get('UNK', 0) + 1
        previous_word = None

        if (index+1 < len(sentence) and index - 1 >= 0):
            word=sentence[index]
            previous_word = sentence[index-1]
            next_word = sentence[index+1]
            bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)+self.calculate_bigram_probabilty(next_word,word)
            return bigram_word_probability


    def log_prob(self, word):
        (w1,w2)=word
        if word in self.bigram_frequencies:
            return math.log(self.bigram_frequencies[word]) - math.log(self.corpus_length)
        else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            return float("-inf")

##
    def check_probs(self):

        # Make sure the probability for each word is between 0 and 1
        for w in self.bigram_frequencies:
            assert 0 - eps < math.exp(self.log_prob(w)) < 1 + eps
        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
            sum([math.exp(self.log_prob(w)) for w in self.bigram_frequencies]) < \
            1 + eps


    
class InterpolateLanguageModel(BigramLanguageModel):
    def __init__(self, file, smoothing=True):
        BigramLanguageModel.__init__(self, file, smoothing=False)

    def calculate_interpolate_probability(self,previous_word, word):
        if word not in self.unigram_frequencies:
            self.unigram_frequencies['UNK']=self.unigram_frequencies.get('UNK', 0) + 1
        bigram_word_probability_numerator = self.bigram_frequencies.get((word,previous_word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(word, 0)

            
        if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0:
            bigram_word_probability=0
        else:
            bigram_word_probability=float(bigram_word_probability_numerator) /  float(bigram_word_probability_denominator)

        
        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        word_probability_numerator += 1
                # add one more to total number of seen unique words for UNK - unseen events
        word_probability_denominator += self.unique_words + 1
        unigram_word_probability=float(word_probability_numerator) / float(word_probability_denominator)
        one_minus_eps=(1-eps)
        prob=(bigram_word_probability - (float(eps)*bigram_word_probability)) + (float(eps)*unigram_word_probability)
        
        return 0.0 if bigram_word_probability == 0 or unigram_word_probability == 0 else math.log(prob)


    def calculate_interpolate_next_probability(self,next_word, word):
        if word not in self.unigram_frequencies:
            self.unigram_frequencies['UNK']=self.unigram_frequencies.get('UNK', 0) + 1
        bigram_word_probability_numerator = self.bigram_frequencies.get((next_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(next_word, 0)

            
        if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0:
            bigram_word_probability=0
        else:
            bigram_word_probability=float(bigram_word_probability_numerator) /  float(bigram_word_probability_denominator)

        
        word_probability_numerator = self.unigram_frequencies.get(next_word, 0)
        word_probability_denominator = self.corpus_length
        word_probability_numerator += 1
                # add one more to total number of seen unique words for UNK - unseen events
        word_probability_denominator += self.unique_words + 1
        unigram_word_probability=float(word_probability_numerator) / float(word_probability_denominator)
        one_minus_eps=(1-eps)
        prob=(bigram_word_probability - eps*bigram_word_probability) + (eps*unigram_word_probability)
        
        return 0.0 if bigram_word_probability == 0 or unigram_word_probability == 0 else math.log(prob)
    

    def calculate_interpolate_word_probability(self,sentence,index):
        if sentence[index] not in self.unigram_frequencies:
                self.unigram_frequencies['UNK']=self.unigram_frequencies.get('UNK', 0) + 1
        previous_word = None

        if (index+1 < len(sentence) and index - 1 >= 0):
            word=sentence[index]
            previous_word = sentence[index-1]
            next_word = sentence[index+1]
            interpolate_word_probability = self.calculate_interpolate_probability(word, previous_word)* self.calculate_interpolate_next_probability(next_word,word)
            return interpolate_word_probability


def edits1(word):

   alphabet= 'abcdefghijklmnopqrstuvwxyz'
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

if __name__ == '__main__':
    import sys

    # Look for the training corpus in the current directory
    train_corpus = 'corpus.txt' 

    # n will be '1', '2' or 'interp' (but this starter code ignores
    # this)
    n = sys.argv[1]

    # The collection of sentences to make predictions for
    predict_corpus = sys.argv[2]


    if n=='1':
        model='Unigram'
        lm = UnigramLanguageModel(train_corpus,smoothing=True)
    elif n=='2':
        model='Bigram'
        lm = BigramLanguageModel(train_corpus,smoothing=True)
    elif n=='interp':
        lm=InterpolateLanguageModel(train_corpus,smoothing=True)

    sorted_vocab_keys = lm.sorted_vocabulary()
    lm.check_probs()
    for line in open(predict_corpus):

        # Split the line on a tab; get the target word to correct and
        # the sentence it's in
        target_index,sentence = line.split('\t')
        target_index = int(target_index)
        sentence = sentence.split()
        target_word = sentence[target_index]

        # Get the in-vocabulary candidates 
        candidates= edits1(target_word)
        iv_candidates = [c for c in candidates if lm.in_vocab(c)   ]

        best_prob = float("-inf")
        best_correction = target_word
        for ivc in iv_candidates:
            sentence[target_index]=ivc
            if n=='1':
                ivc_log_prob = lm.calculate_unigram_probability(ivc)

            elif n=='2':
                ivc_log_prob = lm.calculate_bigram_word_probability(sentence,target_index)

            elif  n=='interp':
                ivc_log_prob = lm.calculate_interpolate_word_probability(sentence,target_index)

            if ivc_log_prob > best_prob:
                best_prob = ivc_log_prob
                best_correction = ivc

        print(best_correction)

