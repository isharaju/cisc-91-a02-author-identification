# CISC 91 - A02 (May 28,2025)
# Group 7 — Isha Raju
## Authorship identification system


🖋️ Stylometric Authorship Identifier

This project uses stylometric analysis — the statistical study of writing style — to identify the likely author of a mystery text by comparing it with known writing samples. It is sub project (A02) as part of a course project for CISC 91.


### 🧠 How It Works

- Feature Extraction: Each known and unknown text is converted into a signature — a vector of stylometric features.
- Weighting: Each feature is assigned a weight based on its importance.
- Similarity Scoring: The unknown signature is compared to each known signature using a weighted distance formula.
- Best Match: The author with the lowest score (most similar style) is selected.

### 🚀 Features

- Extracts a wide range of stylometric features:
- Original textbook features (e.g., average word length, sentence complexity)
- Additional linguistic features (e.g., punctuation density, dialogue ratio, clause complexity)
- Optional POS tag-based features (e.g., noun/verb/adjective ratios)
- Weighted scoring system for comparing writing styles
- Caching for efficient re-analysis of known texts
- Automated test suite to validate each component
- Optional interactive mode to guess authors on the fly


### 🛠 Requirements

Python 3.6+
NLTK (for tokenization and POS tagging)
Only standard Python libraries are used (e.g., os, collections, string)
Install dependencies:
<pre> 
pip install nltk
</pre>
📚 Download required NLTK datasets:
<pre> 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
Pure Python (no external libraries needed except standard os, collections, and string modules)
</pre>
No other third-party libraries are required — the project runs with pure Python and built-in modules.

### 📁 Project Structure

<pre> 
.
├── improved_authorship_identification.py    # Main analysis script
├── known_authors/                           # Directory of known author samples
├── mystery_text/                            # Directory of mystery texts
├── README.md                                # This file
└── .signature_cache.pkl                     # (auto-generated) cache of known signatures
</pre>


### 📌 Usage

1. Add Your Data
- Add known author texts to the known_authors/ directory.
- Add mystery texts to the mystery_text/ directory.

2. By default, all tests are executed when the script is run. If you want to disable test execution, comment out the following lines in the __main__ block:
<pre> 
    run_all_tests() 
</pre>
  
Then run 
<pre> 
python improved_authorship_identification.py
</pre>


3. Make a Guess (non-interactive)
- You can modify the test_make_guess() dictionary with new mystery files and expected authors.

4. Run Interactive Mode
Uncomment the following in the `__main__` block:
<pre> 
make_guess_interactive('known_authors')
</pre>

Then run the script:
<pre> 
python improved_authorship_identification.py
</pre>

🧪 Example Output
<pre> 
File: oliver_twist.txt
Expected: Signature2_Charles_Dickens_A_Tale_of_two_cities.txt
Predicted: Signature2_Charles_Dickens_A_Tale_of_two_cities.txt
✅ PASSED
</pre>


### 🔧 Functions Overview

📚 Text Cleaning and Metrics

| Function                            | Description                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| `clean_word(word)`                  | Cleans a word by removing punctuation and converting it to lowercase. |
| `split_string(text, separators)`    | Splits a string based on custom separator characters.                 |
| `get_sentences(text)`               | Splits text into sentences using `.?!` delimiters.                    |
| `get_phrases(sentence)`             | Splits a sentence into phrases using `,:;` delimiters.                |


📊 Signature & Comparison

| Function                                         | Description                                              |
| ------------------------------------------------ | -------------------------------------------------------- |
| `make_signature(text)`                           | Returns the 5-metric signature of a text.                |
| `get_all_signatures(known_dir)`                  | Builds a signature dictionary for all known authors.     |
| `get_score(sig1, sig2, weights)`                 | Computes the weighted difference between two signatures. |
| `lowest_score(signatures, unknown_sig, weights)` | Finds the closest match in known signatures.             |
| `process_data(mystery_file, known_dir)`          | Identifies the author of a given mystery file.           |


📊 Feature Set

✅ Enabled by Default

| Feature                       | Description                      |
| ----------------------------- | -------------------------------- |
| `average_word_length`         | Mean characters per word         |
| `exactly_once_to_total`       | Ratio of words appearing once    |
| `hapax_legomena_ratio`        | Unique once-used word ratio      |
| `average_sentence_length`     | Words per sentence               |
| `average_sentence_complexity` | Clauses per sentence             |
| `function_word_ratio`         | Use of common function words     |
| `clause_complexity`           | Clause depth estimation          |
| `avg_paragraph_length`        | Sentences per paragraph          |
| `dialogue_ratio`              | Ratio of dialogue in the text    |
| `punctuation_density`         | Punctuation marks per 100 words  |
| `word_length_distribution`    | Average word length distribution |

🧪 Optional POS Features
Set ENABLE_POS_FEATURES = True in the script to enable:
- pos_tag_ratio_JJ (Adjectives)
- pos_tag_ratio_VB (Verbs)
- pos_tag_ratio_PRP (Pronouns)
- pos_tag_ratio_RB (Adverbs)
- pos_tag_ratio_NN (Nouns)
- pos_tag_ratio_IN (Prepositions)


### 🧪 Sample Signature Format

For example, for the enabled features by default, a signature looks like :
<pre> 
[
    4.6,    # average_word_length
    0.15,   # exactly_once_to_total (ratio of words appearing exactly once)
    0.12,   # hapax_legomena_ratio (ratio of unique words)
    12.5,   # average_sentence_length (words per sentence)
    2.3     # average_sentence_complexity (phrases/clauses per sentence)
    4.6,    # average_word_length
    0.15,   # exactly_once_to_total
    0.12,   # hapax_legomena_ratio
    12.5,   # average_sentence_length
    2.3,    # average_sentence_complexity
    0.08,   # function_word_ratio
    1.4,    # clause_complexity
    5.2,    # avg_paragraph_length
    0.25,   # dialogue_ratio
    3.4,    # punctuation_density
    0.55    # word_length_distribution
]
</pre>

If more features are enabled (like POS features), the list length increases accordingly

⚖️ Weights Used in Comparison

The following weights are used to compute the distance between two signatures:

Original features from the textbook : These were received from the textbook

| Function                  | Weight |
| ------------------------- | ------ |
| `average_word_length`     | 11     |
| `exactly_once_to_total`   | 33     |
| `hapax_legomena_ratio`    | 50     |
| `average_sentence_length` | 0.4    |
| `average_sentence_complexity` | 4   |

Newly added features not in the textbook : : These need to be refined based on data driven guidance.

| Function                  | Weight |
| ------------------------- | ------ |
| `function_word_ratio`     | 3      |
| `clause_complexity`       | 0.5    |
| `avg_paragraph_length`    | 3      |
| `dialogue_ratio`          | 1      |
| `punctuation_density`     | 3      |
| `word_length_distribution`| 3      |

### POS (Part-Of-Speech) Features

| Function           | Weight |
| ------------------ | ------ |
| `pos_tag_ratio_JJ` | 6      |
| `pos_tag_ratio_VB` | 5      |
| `pos_tag_ratio_PRP`| 7      |
| `pos_tag_ratio_RB` | 6      |
| `pos_tag_ratio_NN` | 5      |
| `pos_tag_ratio_IN` | 4      |


### ⚡ Caching Mechanism

To improve performance, especially when processing large corpora or repeatedly accessed files, the script uses caching to avoid redundant computations:

| Feature                | Description                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| `signature_cache`      | A dictionary used to store previously computed signatures for known texts. |
| `get_all_signatures()` | Populates `signature_cache` with results, avoiding repeated file reads.    |
| `process_data()`       | Checks cache before re-processing known authors' files.                    |

- Why it matters: Without caching, signature extraction would re-read and re-process all known author files for every mystery text. With caching, we compute once and reuse, resulting in significant speed-up.
- Where it's used: Inside the get_all_signatures() and process_data() functions, where known author texts are read and analyzed.

### 📌 Notes

- Ensure UTF-8 encoding for all text files.
- Empty strings and whitespace-only content are safely handled.
- You can adjust the weights for better tuning depending on the dataset.


### ⚠️ Challenges
Some challenges we faced during development:

- Determining optimal weights for new features

  Assigning meaningful weights manually was challenging. Ideally, this would be automated using machine learning or statistical techniques for better accuracy.

- Performance impact of POS tagging

  Part-of-speech tagging added significant runtime overhead. We had to optimize these functions and make their use optional for practical performance.
  
- Scalability for larger datasets and author pools

  To support more authors and larger corpora, additional performance improvements (e.g., batching, parallelization, persistent caching) may be necessary.

- Extracting more relevant features

  Identifying and engineering stylometric features that meaningfully contribute to author discrimination is non-trivial and requires careful linguistic and statistical insight.


### 📌 Future Enhancements

- Did not get time to write test cases for some functions. Need to do that.
- Need to more robust testing of the functions by adding more complex test cases
- Automate dataset collection.
- Add machine learning models for feature weight learning.
- Incorporate syntactic/semantic metrics.
- Test it on a even more larger corpus of data.
- Currently some of the books (like "great_expectations.txt" and "the_yellow_Wallpaper.txt") were not predicted correctly, with addition of more relevant features and training on more data and selection of better weights can enhance the prediction capability of the system.
- Enable parallelized processing.
- Web-based interface for file input/output
- Integrate visualization for signature comparisons.


### 📚 References

This project was built for CISC 91 - A02, incorporating techniques and concepts from:
- Learn AI assisted Python Programming : With GitHub Copilot and ChatGPT by Leo Porter ● Daniel Zing
- NLTK for natural language processing
- Research in computational linguistics

