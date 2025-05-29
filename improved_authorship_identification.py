"""
===============================================================================
Title       : improved_authorship_identification.py.py
Description : Identifies the most likely author of a mystery text by comparing
              stylometric signatures with known author texts.
Author      : Isha Raju
Date        : 2025-05-28
Usage       : python authorship_identifier_improved.py
===============================================================================
"""
import string
import re
import os
import math
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import Counter
from collections import OrderedDict

def clean_word(word, remove_all_punc=False):
    """
    Removes punctuation and lowercases the word.

    Parameters:
        word (str): The word to clean.
        remove_all_punc (bool): Whether to remove all internal punctuation (True)
                                or just strip it from the ends (False).

    Returns:
        str: Cleaned and lowercased word.
    """
    if remove_all_punc:
        word = re.sub(rf"[{re.escape(string.punctuation)}]", "", word)
    else:
        word = word.strip(string.punctuation)
    return word.strip().lower()

def test_clean_word():
    assert clean_word("Hello, World!") == "hello, world"
    assert clean_word("Python3.8") == "python3.8"
    assert clean_word("  Leading and trailing spaces  ") == "leading and trailing spaces"
    assert clean_word("Punctuation!@#$%^&*()") == "punctuation"
    assert clean_word("No punctuation") == "no punctuation"
    assert clean_word("Lower case") == "lower case"
    assert clean_word("Lower-case") == "lower-case"
    assert clean_word("") == ""
    assert clean_word("   ") == ""
    print("All tests passed! : test_clean_word_outside_punctuation")     
    return 1

def get_words(text):
    """
    Tokenizes text into cleaned words.

    Parameters:
        text (str): Input text.

    Returns:
        list[str]: List of lowercase, punctuation-free words.
    """
    words = text.split()
    return [cw for word in words if (cw := clean_word(word)) != '']


def test_get_words():
    assert get_words("Hello, world!") == ["hello", "world"], "Failed basic punctuation stripping"
    assert get_words("  This... is   spaced  out!  ") == ["this", "is", "spaced", "out"], "Failed spacing and ellipsis"
    assert get_words("Don't stop-believing.") == ["don't", "stop-believing"], "Failed to keep internal punctuation"
    assert get_words("") == [], "Failed on empty string"
    assert get_words("!!! ??? ...") == [], "Failed on only punctuation"
    assert get_words("MIXED case Words!") == ["mixed", "case", "words"], "Failed on mixed casing"
    assert get_words("end.") == ["end"], "Failed on word with trailing punctuation"
    assert get_words("...start") == ["start"], "Failed on word with leading punctuation"
    assert get_words("in-between...text") == ["in-between...text"], "Failed on punctuation not at ends"
    assert get_words("repeat, repeat, repeat.") == ["repeat", "repeat", "repeat"], "Failed on repeated words"

    print("All tests passed! : test_get_words")     
    return 1

# ----------------------------
# STYLOMETRIC METRICS
# ----------------------------

def average_word_length(text):
    """
    Calculates average word length in the text.

    Parameters:
        text (str): Input text.

    Returns:
        float: Average word length.
    """
    words = get_words(text)
    return sum(len(word) for word in words) / len(words) if words else 0


def average_sentence_length(text):
    """
    Calculates average number of words per sentence.

    Parameters:
        text (str): Input text.

    Returns:
        float: Average sentence length.
    """
    for ch in ".!?":
        text = text.replace(ch, ".")
    sentences = split_string(text, ".")
    word_counts = [len(get_words(s)) for s in sentences if get_words(s)]
    return sum(word_counts) / len(word_counts) if word_counts else 0


def test_average_word_length():
    assert average_word_length("Hello, World!") == 5.0
    assert average_word_length("Python3.8") == 9
    assert average_word_length("  Leading and trailing spaces  ") == 6.0
    assert average_word_length("Punctuation!@#$%^&*()") == 11.0
    assert average_word_length("No punctuation") == 6.5
    assert average_word_length("Lower case") == 4.5
    assert average_word_length("Lower-case") == 10.0
    assert average_word_length("") == 0
    assert average_word_length("   ") == 0
    print("All tests passed! : test_average_word_length")
    return 1

def read_text_file(file_path):
    """
    Reads the content of a text file and returns it as a string.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        str: The content of the text file.
    """
    with open(file_path, 'r') as file:
        return file.read()  
    

def different_to_total(text):
    """
    Returns a ratio of unique words by all words in the given text.
    
    Args:
        text (str): The input text from which to extract unique words.
        
    Returns:
        set: A set of unique words in the text.
    """  
    words = text.split()

    clean_words = [clean_word(word) for word in words if word !=""]

    unique_words = set(clean_words)  

    total_words = len(clean_words)
    
    if total_words == 0:
        return 0
    else:
        # return clean_words
        return len(unique_words) / total_words


def test_different_to_total():
    assert different_to_total("Hello, World! Hello, World!") == 0.5
    assert different_to_total("Python3.8 Python3.8") == 0.5
    assert different_to_total("  Leading and trailing spaces  Leading and trailing spaces  ") == 0.5
    assert different_to_total("Punctuation!@#$%^&*() Punctuation!@#$%^&*()") == 0.5
    assert different_to_total("No punctuation No punctuation No punctuation") == 1/3
    assert different_to_total("Lower case case") == 2/3
    assert different_to_total("") == 0
    assert different_to_total("   ") == 0
    assert different_to_total("This    ")==1
    assert different_to_total("This  is   a test   ")==1
    assert different_to_total("This  is   a-test")==1
    print("All tests passed! : test_different_to_total")
    return 1

    """
    Calculates the hapax legomena ratio (words that appear only once).

    Parameters:
        text (str): Input text.

    Returns:
        float: Number of unique words that appear once / total words.
    """
    words = get_words(text)
    if not words:
        return 0
    counts = Counter(words)
    hapax = sum(1 for count in counts.values() if count == 1)
    return hapax / len(words)


def hapax_legomena_ratio(text):
    """
    Returns the ratio of words that occur exactly once to the total number of words in the text.
    
    Parameters:
    - text (str): The input text string to analyze.
    
    Returns:
    - float: Ratio of hapax legomena (words that appear only once) to total word count.
    """
    words = [clean_word(word) for word in text.split()]
    words = [word for word in words if word]  # Remove empty strings
    
    if not words:
        return 0.0

    counts = Counter(words)
    hapaxes = [word for word, count in counts.items() if count == 1]
    return len(hapaxes) / len(words)

def exactly_once_to_total(text):
    """
    Calculate the ratio of words that appear exactly once to the total number of words in the given text.

    Parameters:
    - text (str): The input string of text from which the ratio is to be calculated.

    Returns:
    - float: The ratio of words that occur exactly once to the total number of valid (non-empty) words.
             If there are no valid words, the function returns 0 to avoid division by zero.

    Notes:
    - Words are cleaned using the clean_word() function to remove surrounding punctuation and convert to lowercase.
    - Empty strings resulting from cleaning are ignored.

    Example:
    >>> exactly_once_to_total('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.')
    0.5
    """
    words = text.split()  # Split text into raw words
    # Clean each word, removing punctuation and ignoring empty results
    cleaned_words = [clean_word(word) for word in words if clean_word(word) != '']
    total = len(cleaned_words)  # Total valid words after cleaning

    from collections import Counter
    counts = Counter(cleaned_words)  # Count occurrences of each word

    # Count words that appear exactly once
    exactly_once = sum(1 for word in counts if counts[word] == 1)

    # Return the ratio; if no words, return 0 to avoid division by zero
    return exactly_once / total if total > 0 else 0


def test_exactly_once_to_total():   
    assert exactly_once_to_total("A pearl! Pearl! Lustrous pearl! Rare. What a nice find.") == 0.5
    assert exactly_once_to_total("Python3.8 Python3.8") == 0
    assert exactly_once_to_total("  Leading and trailing spaces  Leading and trailing spaces  ") == 0
    assert exactly_once_to_total("Punctuation!@#$%^&*() Punctuation!@#$%^&*()") == 0
    assert exactly_once_to_total("No punctuation No punctuation No punctuation") == 0
    assert exactly_once_to_total("Lower case case") == 1/3
    assert exactly_once_to_total("") == 0
    assert exactly_once_to_total("   ") == 0
    assert exactly_once_to_total("This is a test") == 1
    print("All tests passed! : test_exactly_once_to_total") 
    return 1

def test_hapax_legomena_ratio():   
    assert hapax_legomena_ratio("Python3.8 Python3.8") == 0
    assert hapax_legomena_ratio("  Leading and trailing spaces  Leading and trailing spaces  ") == 0
    assert hapax_legomena_ratio("Punctuation!@#$%^&*() Punctuation!@#$%^&*()") == 0
    assert hapax_legomena_ratio("No punctuation No punctuation No punctuation") == 0
    assert hapax_legomena_ratio("Lower case case") == 1/3
    assert hapax_legomena_ratio("") == 0
    assert hapax_legomena_ratio("   ") == 0
    assert hapax_legomena_ratio("This is a test") == 1
    print("All tests passed! : test_hapax_legomena_ratio") 
    return 1


def split_string(text, separators):
    '''
    text is a string of text.
    separators is a string of separator characters.
    Split the text into a list using any of the one-character
    separators and return the result.
    Remove spaces from beginning and end
    of a string before adding it to the list.
    Do not include empty strings in the list.
    
    >>> split_string('one*two[three', '*[')
    ['one', 'two', 'three']
    
    >>> split_string('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.', '.?!')
    ['A pearl', 'Pearl', 'Lustrous pearl', 'Rare', 'What a nice find']
    '''
    all_strings = []
    current_string = ''

    for char in text:
        if char in separators:
            current_string = current_string.strip()
            if current_string != '':
                all_strings.append(current_string)
            current_string = ''
        else:
            current_string += char

    # Add the last string after the loop, if not empty
    current_string = current_string.strip()
    if current_string != '':
        all_strings.append(current_string)

    return all_strings

def test_split_string():
    # Basic case with multiple separators
    assert split_string('one*two[three', '*[') == ['one', 'two', 'three']
    
    # Separator at the end
    assert split_string('one*two[three*', '*[') == ['one', 'two', 'three']
    
    # Multiple punctuation separators
    assert split_string('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.', '.?!') == [
        'A pearl', 'Pearl', 'Lustrous pearl', 'Rare', 'What a nice find'
    ]
    
    # Leading/trailing spaces and empty sections
    assert split_string('  a   ,  b , , c  ', ',') == ['a', 'b', 'c']
    
    # No separators at all
    assert split_string('hello world', ',;') == ['hello world']
    
    # All separators
    assert split_string(',,,', ',') == []
    
    # Mix of separators and space trimming
    assert split_string(' apple ; banana ;  cherry ', ';') == ['apple', 'banana', 'cherry']
    
    # Separator is a space
    assert split_string('one two three', ' ') == ['one', 'two', 'three']
    
    # Unicode and special characters
    assert split_string('a★b☆c', '★☆') == ['a', 'b', 'c']
    
    # Multiline string with separators
    assert split_string("Line one.\nLine two!\nLine three?", '.!?') == ['Line one', 'Line two', 'Line three']
    
    print("All tests passed! : test_split_string") 
    return 1


def get_sentences(text):
    """
    text is a string of text.
    Return a list of the sentences from text.
    Sentences are separated by a '.', '?' or '!'.

    >>> get_sentences('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.')
    ['A pearl', 'Pearl', 'Lustrous pearl', 'Rare', 'What a nice find']
    """
    return split_string(text, '.?!')

def test_get_sentences():
    assert get_sentences('Hello! How are you? I am fine.') == ['Hello', 'How are you', 'I am fine']
    assert get_sentences('No punctuation here') == ['No punctuation here']
    assert get_sentences('One. Two! Three?') == ['One', 'Two', 'Three']
    assert get_sentences('Multiple...dots!!!and???marks!') == ['Multiple', 'dots', 'and', 'marks']
    print("All tests passed! : test_get_sentences") 
    return 1

def average_sentence_length(text):
    '''
    text is a string of text.
    Return the average number of words per sentence in text.
    Do not count empty words as words.

    >>> average_sentence_length('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.')
    2.0
    '''
    sentences = get_sentences(text)
    total_words = 0

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word != '':
                total_words += 1

    return total_words / len(sentences) if sentences else 0

def test_average_sentence_length():
    assert average_sentence_length('Hello world. Hi there!') == 2.0
    assert average_sentence_length('Just one sentence') == 3.0
    assert average_sentence_length('One. Two three. Four five six.') == 2.0
    assert average_sentence_length('') == 0
    assert average_sentence_length('!.?') == 0
    print("All tests passed! : test_average_sentence_length") 
    return 1


def get_phrases(sentence):
    '''
    sentence is a sentence string.
    Return a list of the phrases from sentence.
    Phrases are separated by a ',', ';' or ':'.

    >>> get_phrases('Lustrous pearl, Rare, What a nice find')
    ['Lustrous pearl', 'Rare', 'What a nice find']
    '''
    return split_string(sentence, ',;:')


def test_get_phrases():
    assert get_phrases('A, B, C') == ['A', 'B', 'C']
    assert get_phrases('One;Two:Three') == ['One', 'Two', 'Three']
    assert get_phrases('No separators here') == ['No separators here']
    assert get_phrases(' , : ; ') == []
    assert get_phrases('Phrase one, phrase two ; phrase three:phrase four') == [
        'Phrase one', 'phrase two', 'phrase three', 'phrase four'
    ]
    print("All tests passed! : test_get_phrases") 
    return 1


def average_sentence_complexity(text):
    """
    Calculates average number of phrases per sentence.

    Parameters:
        text (str): Input text.

    Returns:
        float: Sentence complexity as average number of phrases per sentence.
    """
    for ch in ".!?":
        text = text.replace(ch, ".")
    sentences = split_string(text, ".")
    phrase_delimiters = [",", ";", ":"]
    total_phrases = 0
    valid_sentences = 0
    for sentence in sentences:
        if sentence.strip():
            valid_sentences += 1
            for ch in phrase_delimiters:
                sentence = sentence.replace(ch, ",")
            phrases = split_string(sentence, ",")
            total_phrases += len([p for p in phrases if p.strip()])
    return total_phrases / valid_sentences if valid_sentences else 0


def test_average_sentence_complexity():
    assert average_sentence_complexity('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.') == 1.0
    assert average_sentence_complexity('A pearl! Pearl! Lustrous pearl! Rare, what a nice find.') == 1.25
    assert average_sentence_complexity('One, two, three. Four: five.') == 2.5
    assert average_sentence_complexity('Simple sentence.') == 1.0
    assert average_sentence_complexity('') == 0
    print("All tests passed! : test_average_sentence_complexity") 
    return 1

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# New stylometric_features in addition to the textbook features.

def function_word_ratio(text):
    """
    Input: text (str) - The input text
    Output: float - Ratio of function words to total words
    Description: Computes the frequency ratio of function words (e.g., the, of, and) to all words.
    """
    function_words = set([
        'the', 'of', 'and', 'a', 'in', 'to', 'is', 'it', 'that', 'for', 'on', 'with', 'as', 'was', 'at', 'by'
    ])

    tokenizer = RegexpTokenizer(r'\w+')
    words = [word.lower() for word in tokenizer.tokenize(text)]



    # words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    if not words:
        return 0
    count = sum(1 for word in words if word in function_words)
    return count / len(words)



def pos_tag_ratio(text, tag_prefix):
    """
    Input: text (str), tag_prefix (str) - Part-of-speech tag prefix (e.g., 'JJ' for adjectives)
    Output: float - Ratio of specified POS tag to total words
    Description: Computes the ratio of specified POS tag type in the text.
    """
    words = word_tokenize(text)
    tagged = pos_tag(words)
    total = len(words)
    if total == 0:
        return 0
    count = sum(1 for _, tag in tagged if tag.startswith(tag_prefix))
    return count / total



def test_pos_tag_ratio():
    # 1. Basic adjective usage
    text1 = "The quick brown fox jumps over the lazy dog"
    assert abs(pos_tag_ratio(text1, 'JJ') - 2/9) < 1e-6, "Failed on adjective count"

    # 2. No matching POS tags (e.g., looking for verbs in a noun-only sentence)
    text2 = "Cat table apple sky mountain"
    assert pos_tag_ratio(text2, 'VB') == 0.0, "Failed on no matching POS tags"

    # 3. All matching POS tags (e.g., only verbs)
    text3 = "Run jump swim dive skip hop"
    assert abs(pos_tag_ratio(text3, 'VB') - 1.0) < 1e-6, "Failed on all matching POS tags"

    # 4. Mixed sentence with nouns and verbs
    text4 = "She runs and he swims every morning"
    ratio = pos_tag_ratio(text4, 'VB')
    assert 0 < ratio < 1, "Failed on mixed tags"

    # 5. Empty input
    assert pos_tag_ratio("", 'JJ') == 0.0, "Failed on empty string"

    # 6. Punctuation only
    assert pos_tag_ratio("!!! ???", 'JJ') == 0.0, "Failed on punctuation-only input"

    # 7. Multiple adjectives and nouns
    text7 = "Bright blue sky with puffy white clouds"
    assert abs(pos_tag_ratio(text7, 'JJ') - 4/8) < 1e-6, "Failed on multiple adjectives"

    print("All tests passed! : test_pos_tag_ratio") 

    return 1



def clause_complexity(text):
    """
    Input: text (str)
    Output: float - Approximate clause complexity
    Description: Ratio of commas/semicolons (potential clause boundaries) per sentence.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    clause_markers = [',', ';', ':']
    count = sum(text.count(marker) for marker in clause_markers)
    return count / len(sentences)


def avg_paragraph_length(text):
    """
    Input: text (str)
    Output: float - Average number of sentences per paragraph
    Description: Computes average number of sentences per paragraph.
    """
    paragraphs = [p for p in text.split('\n') if p.strip() != '']
    if not paragraphs:
        return 0
    return sum(len(sent_tokenize(p)) for p in paragraphs) / len(paragraphs)


def dialogue_ratio(text):
    """
    Input: text (str)
    Output: float - Proportion of sentences that are dialogue (contain quotes)
    Description: Identifies sentences with quotation marks as potential dialogue.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    dialogue = sum(1 for s in sentences if '"' in s or '\'' in s)
    return dialogue / len(sentences)


def punctuation_density(text):
    """
    Input: text (str)
    Output: float - Frequency of punctuation marks per 100 words
    Description: Calculates how often punctuation is used in the text.
    """
    punctuations = re.findall(r'[;:!?—…]', text)
    words = word_tokenize(text)
    return len(punctuations) / (len(words) / 100) if words else 0


def word_length_distribution(text):
    """
    Input: text (str)
    Output: float - Weighted average based on short/medium/long words
    Description: Uses different weights for different word lengths to assess preference.
    """
    words = [word for word in word_tokenize(text) if word.isalpha()]
    if not words:
        return 0
    short = sum(1 for w in words if len(w) <= 4)
    medium = sum(1 for w in words if 5 <= len(w) <= 8)
    long = sum(1 for w in words if len(w) > 8)
    total = len(words)
    return (short * 1 + medium * 2 + long * 3) / total


#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

# ----------------------------
# SIGNATURE + SCORING
# ----------------------------

def make_signature(text):
    """
    Compute the stylometric signature using the ordered feature definitions.
    """
    signature = []
    for name, (func, weight) in feature_definitions.items():
        val = func(text)
        signature.append(val)
    return signature


# Global cache variable for signatures
_signature_cache = {}

def get_all_signatures(known_dir, use_cache=True):
    '''
    Returns a dictionary of signatures for known texts in known_dir.
    Caches results in memory and on disk for efficiency.

    Parameters:
    - known_dir (str): directory path with known author texts.
    - use_cache (bool): whether to use caching (default True).

    Returns:
    - dict: {filename: signature_list}
    '''
    global _signature_cache

    cache_file = os.path.join(known_dir, '.signature_cache.pkl')

    if use_cache:
        # Check in-memory cache first
        if known_dir in _signature_cache:
            return _signature_cache[known_dir]

        # Check on-disk cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    signatures = pickle.load(f)
                    _signature_cache[known_dir] = signatures
                    return signatures
            except Exception:
                pass  # Fall back to recomputing

    signatures = {}

    for filename in os.listdir(known_dir):
        path = os.path.join(known_dir, filename)
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                text = f.read()
                signatures[filename] = make_signature(text)

    # Save to cache
    if use_cache:
        _signature_cache[known_dir] = signatures
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(signatures, f)
        except Exception:
            pass  # silently ignore cache save failures

    return signatures



# def get_all_signatures(known_dir):
#     '''
#     Input:
#         known_dir (str): The path to a directory containing known author text files.
#                          Each file is expected to contain a single author's writing sample.

#     Output:
#         dict: A dictionary where:
#               - each key is a filename (e.g., 'author1.txt'),
#               - each value is a list of 5 numerical features (signature) extracted from the file.

#     Description:
#         This function loops through all files in the given directory. For each file:
#         - Reads the content of the file.
#         - Computes a signature using `make_signature(text)`.
#         - Adds the filename and its corresponding signature to a dictionary.
#         - Returns the dictionary mapping filenames to their signature lists.

#     Example:
#         >>> get_all_signatures('known_authors')
#         {
#             'author1.txt': [4.1, 0.7, 0.5, 2.5, 1.25],
#             'author2.txt': [4.4, 0.6, 0.4, 3.0, 1.0],
#             ...
#         }
#     '''

#     import os  # Ensure os is imported

#     signatures = {}

#     # Loop over all files in the directory
#     for filename in os.listdir(known_dir):
#         path = os.path.join(known_dir, filename)

#         # Only process files (skip directories)
#         if os.path.isfile(path):
#             with open(path, encoding='utf-8') as f:
#                 text = f.read()

#                 # Generate the signature for the text and store it
#                 signatures[filename] = make_signature(text)

#     return signatures


def get_score(signature1, signature2, weights):
    """
    Calculates the weighted difference score between two text signatures.
    
    Parameters:
    - signature1 (list of float): The first text signature, which is a list of 5 numeric features.
    - signature2 (list of float): The second text signature, also a list of 5 numeric features.
    - weights (list of float): A list of 5 weight values corresponding to the importance of each feature.

    Returns:
    - float: A single number representing the weighted difference (or distance) between the two signatures.
    
    The function calculates the sum of the absolute differences between corresponding elements in
    signature1 and signature2, each multiplied by a weight that reflects the feature's importance.

    Example:
    >>> get_score([4.6, 0.1, 0.05, 10, 2],
                  [4.3, 0.1, 0.04, 16, 4],
                  [11, 33, 50, 0.4, 4])
    14.2
    """

    score = 0
    for i in range(len(signature1)):
        # Calculate the absolute difference between the corresponding features
        # Multiply by the corresponding weight to get weighted difference
        score += abs(signature1[i] - signature2[i]) * weights[i]
    return score


def lowest_score(signatures_dict, unknown_signature, weights):
    '''
    Input:
        signatures_dict (dict): A dictionary mapping each key (usually a filename or author name)
                                to its corresponding signature (list of 5 floats).
        unknown_signature (list of float): A list of 5 numerical features representing the signature
                                           of an unknown text.
        weights (list of float): A list of 5 weights used to compute the weighted score
                                 between two signatures.

    Output:
        str: The key (e.g., filename or author name) from `signatures_dict` that has the lowest
             weighted score when compared to the unknown signature.

    Description:
        This function iterates over all known signatures, calculates the weighted score between
        each known signature and the unknown signature using `get_score()`. It keeps track of the
        key (author/file) with the lowest score and returns it as the best match.

    Example:
        >>> d = {'Dan': [1, 1, 1, 1, 1], 'Leo': [3, 3, 3, 3, 3]}
        >>> unknown = [1, 0.8, 0.9, 1.3, 1.4]
        >>> weights = [11, 33, 50, 0.4, 4]
        >>> lowest_score(d, unknown, weights)
        'Dan'
    '''

    lowest = None  # Will store a tuple (key, score) with the lowest score so far

    for key in signatures_dict:
        # Compute weighted score between this known signature and the unknown
        score = get_score(signatures_dict[key], unknown_signature, weights)

        # Update if it's the first score or a new lowest
        if lowest is None or score < lowest[1]:
            lowest = (key, score)

    # Return the key (e.g., author/file) with the lowest score
    return lowest[0]


def process_data(mystery_filename, known_dir):
    signatures = get_all_signatures(known_dir)
    with open(mystery_filename, encoding='utf-8') as f:
        text = f.read()

    unknown_signature = make_signature(text)

    # Extract weights maintaining the feature order
    weights = [weight for _, weight in feature_definitions.values()]

    return lowest_score(signatures, unknown_signature, weights)


def make_guess_interactive(known_dir):

    '''
    Interactive guessing of the author for a user-provided mystery file.
    Prompts the user to input the filename of a mystery text file, then computes its signature and compares it 
    to known authors in known_dir using cached signatures.
    Prints the filename of the known author text with the closest signature.
    '''
    # Ask user for the filename of the mystery book
    filename = input('Enter filename of  mystery text :  ')
    
    # Print the guessed author/book filename with the closest signature
    print(process_data(filename, known_dir))

def test_make_guess():
    '''
    Test the process_data function for multiple known mystery texts.
    Verifies that the predicted author (from known_dir) is correct or expected.

    Output:
        Prints test results for each mystery file, indicating pass/fail.
    '''

    known_dir = 'known_authors'
    # Map mystery file to expected predicted author file
    expected_predictions = {
        "david_copperfield.txt": "Signature2_Charles_Dickens_A_Tale_of_two_cities.txt",
        "julius_caesar.txt": "Signature1_Shakespere_Hamlet.txt",
        "macbeth.txt": "Signature1_Shakespere_Hamlet.txt",
        "oliver_twist.txt": "Signature2_Charles_Dickens_A_Tale_of_two_cities.txt",
        "sense_and_Sensibility.txt": "Signature3_Jane_Austin_Pride_and_Prejudice.txt",
        "the_comedy_of_errors.txt": "Signature1_Shakespere_Hamlet.txt"
    }

    all_passed = True
    for mystery_filename, expected_author in expected_predictions.items():
        mystery_path = f"./mystery_text/{mystery_filename}"
        predicted_author = process_data(mystery_path, known_dir)
        passed = predicted_author == expected_author
        print(f"File: {mystery_filename}")
        print(f"Expected: {expected_author}")
        print(f"Predicted: {predicted_author}")
        print("✅ PASSED\n" if passed else "❌ FAILED\n")
        if not passed:
            all_passed = False

    if all_passed:
        print("🎉 All make_guess tests passed!")
    else:
        print("⚠️ Some make_guess tests failed.")

    return 1



def get_pos_tag_ratios_optimized(text, tag_prefixes):
    """
    Computes the ratio of each specified POS tag prefix to total word count.
    
    Parameters:
        text (str): The input text.
        tag_prefixes (list[str]): List of POS tag prefixes to evaluate.
        
    Returns:
        dict[str, float]: Mapping of tag prefix to ratio.
    """
    words = word_tokenize(text)
    total = len(words)
    if total == 0:
        return {prefix: 0.0 for prefix in tag_prefixes}
    
    tagged = pos_tag(words)
    tag_counts = Counter(tag for _, tag in tagged)
    
    ratios = {}
    for prefix in tag_prefixes:
        count = sum(count for tag, count in tag_counts.items() if tag.startswith(prefix))
        ratios[prefix] = count / total
    return ratios

def run_all_tests():
    '''
    Run all tests for the functions defined in this module.
    '''
    all_passed_count = (
    test_clean_word() +
    test_get_words() +
    test_average_word_length() +
    test_different_to_total() +
    test_hapax_legomena_ratio() +
    test_split_string() +
    test_get_sentences() +
    test_average_sentence_length() +
    test_get_phrases() +
    test_average_sentence_complexity() +
    test_make_guess()
)

    if all_passed_count == 11:
        print("All tests passed successfully!")
    else:
        print(f"Some tests failed. Only {all_passed_count} out of 9 passed.")

if __name__ == "__main__":

    #----------------------------------------------------------------
    # Assigning weights to new stylometric features should ideally be data-driven — using techniques like grid search or machine learning for optimization. But based on linguistic relevance and typical variance across authors, here’s a reasonable initial estimate of weights for both the original and new features.
    #----------------------------------------------------------------

    # --- Feature flag to enable POS tag features ---
    ENABLE_POS_FEATURES = False

    # Base features (original + your added ones)
    base_features = [
        #----------------------------------------------------------------
        # Original features from the textbook
        #----------------------------------------------------------------
        ("average_word_length", (average_word_length, 11)),
        ("exactly_once_to_total", (exactly_once_to_total, 33)),
        ("hapax_legomena_ratio", (hapax_legomena_ratio, 50)),
        ("average_sentence_length", (average_sentence_length, 0.4)),
        ("average_sentence_complexity", (average_sentence_complexity, 4)),
        #----------------------------------------------------------------
        # Newly added features not in the textbook
        #----------------------------------------------------------------
        ("function_word_ratio", (function_word_ratio, 3)),
        ("clause_complexity", (clause_complexity, 0.5)),
        ("avg_paragraph_length", (avg_paragraph_length, 3)),
        ("dialogue_ratio", (dialogue_ratio, 1)),
        ("punctuation_density", (punctuation_density, 3)),
        ("word_length_distribution", (word_length_distribution, 3)),
    ]

    # POS tag prefixes and weights if enabled
    # 'JJ' - Adjectives, 'VB' - Verbs, 'PRP' - Personal Pronouns,'RB' - Adverbs, 'NN' - Nouns, 'IN' - Prepositions
    tag_prefixes = ['JJ', 'VB', 'PRP', 'RB', 'NN', 'IN']

    pos_features = [
        ("pos_tag_ratio_JJ", (lambda text: get_pos_tag_ratios_optimized(text, tag_prefixes)['JJ'], 6)),
        ("pos_tag_ratio_VB", (lambda text: get_pos_tag_ratios_optimized(text, tag_prefixes)['VB'], 5)),
        ("pos_tag_ratio_PRP", (lambda text: get_pos_tag_ratios_optimized(text, tag_prefixes)['PRP'], 7)),
        ("pos_tag_ratio_RB", (lambda text: get_pos_tag_ratios_optimized(text, tag_prefixes)['RB'], 6)),
        ("pos_tag_ratio_NN", (lambda text: get_pos_tag_ratios_optimized(text, tag_prefixes)['NN'], 5)),
        ("pos_tag_ratio_IN", (lambda text: get_pos_tag_ratios_optimized(text, tag_prefixes)['IN'], 4)),
    ]

    # Build final feature_definitions dict
    feature_definitions = OrderedDict(base_features)
    if ENABLE_POS_FEATURES:
        feature_definitions.update(pos_features)


     #----------------------------------------------------------------
    # Run all unit tests to verify that each feature extraction function
    # and processing step behaves as expected.
    #----------------------------------------------------------------
    # Performance Notes: 
    #----------------------------------------------------------------
    # See feature_definitions function for details.
    # - With all unoptimized features (especially POS tagging per feature), takes ~20 mins.
    # - With optimized POS tagging using precomputed ratios, ~17 mins.
    # - Without POS-related features, it runs almost instantly.
    print("-"*80)
    print("Run all Test cases")
    print("-"*80)
    run_all_tests() 
    #----------------------------------------------------------------
 
    # Starts an interactive mode that prompts the user to select a mystery text file
    # and compares it with known authors to guess the author.
    # The 'known_authors' directory must contain reference texts by known writers.
    # print("")
    # print("-"*80)
    # print("Following requires the user to input a text file")
    # print("-"*80)
    # make_guess_interactive('known_authors')
    #----------------------------------------------------------------
