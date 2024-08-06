import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
def nxt_word(prev_word, top_n=3):
    possibilities = [b for b in b_count.keys() if b[0] == prev_word]
    sorted_bs = sorted(possibilities, key=lambda b: b_probs[b], reverse=True)
    return [b[1] for b in sorted_bs[:top_n]]

txt = 'The recent economic summit in Geneva brought together global leaders to discuss pressing issues such as climate change, international trade, and sustainable development. Among the significant outcomes was the commitment of several nations to ambitious carbon reduction targets and the establishment of a new trade agreement aimed at reducing tariffs and fostering economic cooperation. Meanwhile, advancements in medical research have led to a breakthrough in cancer treatment, with a new drug developed by scientists at the National Institute of Health showing promising results in clinical trials. This drug targets cancer cells more precisely, significantly increasing survival rates among patients with previously untreatable forms of cancer. These developments highlight the ongoing efforts to address critical global challenges and improve the quality of life through scientific innovation and international collaboration.'
tokens = word_tokenize(txt.lower())
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("Unigrams are:\n", unigrams)
print('\n')
print("Bigrams are:\n", bigrams)
print('\n')
print("Trigrams are:\n", trigrams)
print('\n')
u_count = Counter(unigrams)
b_count = Counter(bigrams)
t_count = Counter(trigrams)

print("\nUnigram Counts:\n", u_count)
print("\nBigram Counts:\n", b_count)
print("\nTrigram Counts:\n", t_count)

total_b = sum(b_count.values())
b_probs = {b: c/total_b for b, c in b_count.items()}
print("\nBigram Probability:")
for b, prob in b_probs.items():
    print(b, " ", prob)

prev_word = input("Enter the previous word: ")
predicted_words = nxt_word(prev_word)
print("Next word prediction for '" + prev_word + "': " + str(predicted_words))