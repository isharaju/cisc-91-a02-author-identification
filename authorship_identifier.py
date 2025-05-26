import string
import re
import os

def clean_word(word, remove_all_punc=False):
    """
    Cleans a word by removing punctuation and converting it to lowercase.           
    For example, "Hello, World!" becomes "hello world".    

    input: word (str): The word to clean.
    output: str: The cleaned word.                     
    """ 
    if remove_all_punc:
        word = remove_all_punc(word)
    else:
        word = word.strip(string.punctuation)
        
    word = word.strip()
    word = word.lower()
    return word 

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


def average_word_length(text, debug = False):
    '''
    text is a string of text.
    Return the average word length of the words in text.
    Do not count empty words as words.
    Do not include surrounding punctuation  
    '''

    words = text.split()
    words = [clean_word(word) for word in words]
    if not words:
        return 0
    total_length = sum(len(word) for word in words)
    for word in words:
        if debug:
            print(f"Word: {word}, Length: {len(word)}")
            
    return total_length / len(words)

def test_average_word_length():
    assert average_word_length("Hello, World!") == 5.0
    assert average_word_length("Python3.8") == 9
    # assert average_word_length("  Leading and trailing spaces  ", debug = True) == 6.0
    assert average_word_length("  Leading and trailing spaces  ") == 6.0
    assert average_word_length("Punctuation!@#$%^&*()") == 11.0
    assert average_word_length("No punctuation") == 6.5
    assert average_word_length("Lower case") == 4.5
    assert average_word_length("Lower-case") == 10.0
    assert average_word_length("") == 0
    assert average_word_length("   ") == 0
    print("All tests passed! : test_average_word_length")

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
     # different_to_total("!@#$%^&*()") 
    # different_to_total("!@#$%^&*()") 
    print("All tests passed! : test_different_to_total")


def exactly_once_to_total(text):
    '''
    text is a string of text.
    Return the number of words that show up exactly once in text
    divided by the total number of words in text.
    Do not count empty words as words.
    Do not include surrounding punctuation.
    >>> exactly_once_to_total('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.')
    0.5
    '''
    words = text.split()
    cleaned_words = [clean_word(word) for word in words if clean_word(word) != '']
    total = len(cleaned_words)

    from collections import Counter
    counts = Counter(cleaned_words)
    exactly_once = sum(1 for word in counts if counts[word] == 1)

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

def average_sentence_complexity(text):
    '''
    text is a string of text.
    Return the average number of phrases per sentence in text.

    >>> average_sentence_complexity('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.')
    1.0
    >>> average_sentence_complexity('A pearl! Pearl! Lustrous pearl! Rare, what a nice find.')
    1.25
    '''
    sentences = get_sentences(text)
    total_phrases = 0

    for sentence in sentences:
        phrases = get_phrases(sentence)
        total_phrases += len(phrases)

    return total_phrases / len(sentences) if sentences else 0

def test_average_sentence_complexity():
    assert average_sentence_complexity('A pearl! Pearl! Lustrous pearl! Rare. What a nice find.') == 1.0
    assert average_sentence_complexity('A pearl! Pearl! Lustrous pearl! Rare, what a nice find.') == 1.25
    assert average_sentence_complexity('One, two, three. Four: five.') == 2.5
    assert average_sentence_complexity('Simple sentence.') == 1.0
    assert average_sentence_complexity('') == 0
    print("All tests passed! : test_average_sentence_complexity") 




def make_signature(text):
    '''
    The signature for text is a list of five elements:
    1. Average word length
    2. Different words divided by total words
    3. Words used exactly once divided by total words
    4. Average sentence length
    5. Average sentence complexity

    Return the signature for text.

    >>> make_signature('A pearl! Pearl! Lustrous pearl! Rare, what a nice find.')
    [4.1, 0.7, 0.5, 2.5, 1.25]
    '''
    return [
        average_word_length(text),
        different_to_total(text),
        exactly_once_to_total(text),
        average_sentence_length(text),
        average_sentence_complexity(text)
    ]
import os

def get_all_signatures(known_dir):
    '''
    known_dir is the name of a directory of books.
    For each file in directory known_dir, determine its signature.
    Return a dictionary where each key is the name of a file,
    and the value is its signature.

    >>> get_all_signatures('known_authors')
    {
        'author1.txt': [4.1, 0.7, 0.5, 2.5, 1.25],
        'author2.txt': [4.4, 0.6, 0.4, 3.0, 1.0],
        ...
    }
    '''
    signatures = {}
    for filename in os.listdir(known_dir):
        path = os.path.join(known_dir, filename)
        if os.path.isfile(path):  # Ensure it's a file
            with open(path, encoding='utf-8') as f:
                text = f.read()
                signatures[filename] = make_signature(text)
    return signatures


def get_score(signature1, signature2, weights):
    '''
    signature1 and signature2 are signatures (lists of five floats).
    weights is a list of five weights (floats).
    Return the weighted score comparing signature1 and signature2.

    >>> get_score([4.6, 0.1, 0.05, 10, 2],
                  [4.3, 0.1, 0.04, 16, 4],
                  [11, 33, 50, 0.4, 4])
    14.2
    '''
    score = 0
    for i in range(len(signature1)):
        # Calculate weighted absolute difference per element
        score += abs(signature1[i] - signature2[i]) * weights[i]
    return score

def lowest_score(signatures_dict, unknown_signature, weights):
    '''
    signatures_dict is a dictionary mapping keys to signatures.
    unknown_signature is a signature.
    weights is a list of five weights.
    Return the key whose signature value has the lowest
    score with unknown_signature.

    >>> d = {'Dan': [1, 1, 1, 1, 1], 'Leo': [3, 3, 3, 3, 3]}
    >>> unknown = [1, 0.8, 0.9, 1.3, 1.4]
    >>> weights = [11, 33, 50, 0.4, 4]
    >>> lowest_score(d, unknown, weights)
    'Dan'
    '''
    lowest = None
    for key in signatures_dict:
        score = get_score(signatures_dict[key], unknown_signature, weights)
        if lowest is None or score < lowest[1]:
            lowest = (key, score)
    return lowest[0]


def process_data(mystery_filename, known_dir):
    '''
    mystery_filename is the filename of a mystery book whose
    author we want to know.
    known_dir is the name of a directory of books.
    Return the name of the signature closest to
    the signature of the text of mystery_filename.

    >>> process_data('unknown1.txt', 'known_authors')
    'Arthur_Conan_Doyle.txt'
    '''
    # Get signatures of all known authors/books
    signatures = get_all_signatures(known_dir)
    
    # Read the mystery book text
    with open(mystery_filename, encoding='utf-8') as f:
        text = f.read()
    
    # Compute the signature of the mystery book
    unknown_signature = make_signature(text)
    
    # Use previously defined weights
    weights = [11, 33, 50, 0.4, 4]
    
    # Return the known author/book whose signature is closest
    return lowest_score(signatures, unknown_signature, weights)

def make_guess_interactive(known_dir):
    '''
    Ask user for a filename.
    Get all known signatures from known_dir,
    and print the name of the one that has the lowest score
    with the user's filename.
    '''
    # Ask user for the filename of the mystery book
    filename = input('Enter filename: ')
    
    # Print the guessed author/book filename with the closest signature
    print(process_data(filename, known_dir))

def make_guess_non_interactive(known_dir):
    '''
    Ask user for a filename.
    Get all known signatures from known_dir,
    and print the name of the one that has the lowest score
    with the user's filename.
    '''
    filename_list = [
                    # Predicted correctly
                     "david_copperfield.txt",
                     "julius_caesar.txt",
                     "macbeth.txt",
                     "oliver_twist.txt",
                     "sense_and_Sensibility.txt",
                     "the_comedy_of_errors.txt",

                     # Did not Predict correctly
                     "great_expectations.txt",
                     "the_yellow_wallpaper.txt"]


    for filename in filename_list:
        mystery_path = f"./mystery_text/{filename}"
        print(f"\nMystery File: {filename}")
        print("Predicted Author:", process_data(mystery_path, known_dir))

if __name__ == "__main__":

    print("-"*80)
    print("Test cases")
    print("-"*80)
    test_clean_word()
    test_average_word_length()
    test_different_to_total()
    test_exactly_once_to_total()
    test_split_string()
    test_get_sentences()
    test_average_sentence_length()
    test_get_phrases()
    test_average_sentence_complexity()
    print("")
    print("-"*80)
    print("Printing result for some books")
    print("-"*80)
    make_guess_non_interactive('known_authors')
    print("")
    print("-"*80)
    print("Following requires the user to input a text file")
    print("-"*80)
    make_guess_interactive('known_authors')
