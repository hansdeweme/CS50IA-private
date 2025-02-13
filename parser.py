import nltk
import sys
import re

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
Modal -> "can" | "must" | "should"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | NP VP Conj VP
AP -> Adj | Adj AP
NP -> N | Det NP | AP NP | NP PP
PP -> P NP | P S
VP -> V | Modal V | V NP | V NP PP | V PP | VP Adv | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    s = preprocess(s)

    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    for tree in trees:
        tree.pretty_print()
        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))

def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)
    return [word for word in words if re.match('[a-z]', word)]

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    return [subtree for subtree in tree.subtrees(is_np_chunk)]

def is_np_chunk(tree):
    """
    Returns true if given tree is a NP chunk.
    """
    return tree.label() == 'NP' and \
           not list(tree.subtrees(lambda t: t.label() == 'NP' and t != tree))

if __name__ == "__main__":
    main()
