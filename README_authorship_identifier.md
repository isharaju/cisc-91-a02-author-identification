# cisc-91-a02-author-identification

🧠 Text Signature Analyzer

This project is a stylometric analysis tool that calculates a "signature" of a text — a set of statistical metrics that reflect the writing style of an author — and compares it to known samples to guess authorship.

📋 Features

- Clean and normalize text data
- Calculate linguistic signatures based on:
- Average word length
- Lexical diversity (unique/total words)
- Hapax legomena ratio (words used exactly once/total words)
- Average sentence length
- Sentence complexity (average phrases per sentence)
- Compare texts using weighted scoring
- Identify the most stylistically similar known author for a mystery text
- Includes both interactive and non-interactive modes for author prediction

🗃️ Project Structure

```
.
├── main.py (or your_script.py)
├── known_authors/
│   ├── author1.txt
│   ├── author2.txt
│   └── ...
└── mystery_texts/
    ├── unknown1.txt
    └── ...

```

🔧 Functions Overview

📚 Text Cleaning and Metrics

| Function                            | Description                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| `clean_word(word)`                  | Cleans a word by removing punctuation and converting it to lowercase. |
| `average_word_length(text)`         | Returns the average length of words in the text.                      |
| `different_to_total(text)`          | Ratio of unique words to total words.                                 |
| `exactly_once_to_total(text)`       | Ratio of words used exactly once to total words.                      |
| `split_string(text, separators)`    | Splits a string based on custom separator characters.                 |
| `get_sentences(text)`               | Splits text into sentences using `.?!` delimiters.                    |
| `get_phrases(sentence)`             | Splits a sentence into phrases using `,:;` delimiters.                |
| `average_sentence_length(text)`     | Average number of words per sentence.                                 |
| `average_sentence_complexity(text)` | Average number of phrases per sentence.                               |


📊 Signature & Comparison

| Function                                         | Description                                              |
| ------------------------------------------------ | -------------------------------------------------------- |
| `make_signature(text)`                           | Returns the 5-metric signature of a text.                |
| `get_all_signatures(known_dir)`                  | Builds a signature dictionary for all known authors.     |
| `get_score(sig1, sig2, weights)`                 | Computes the weighted difference between two signatures. |
| `lowest_score(signatures, unknown_sig, weights)` | Finds the closest match in known signatures.             |
| `process_data(mystery_file, known_dir)`          | Identifies the author of a given mystery file.           |


🧪 Testing & CLI

| Function                                | Description                                             |
| --------------------------------------- | ------------------------------------------------------- |
| `test_*()`                              | Test functions to validate logic.                       |
| `make_guess_interactive(known_dir)`     | Prompts user to input a file and identifies its author. |
| `make_guess_non_interactive(known_dir)` | Loops through predefined filenames for batch analysis.  |

🧪 Sample Signature Format
<pre> 
[avg_word_length, unique/total_words, once/total_words, avg_sentence_len, sentence_complexity]

[4.3, 0.6, 0.4, 10.0, 1.25]
</pre>

⚖️ Weights Used in Comparison

The following weights are used to compute the distance between two signatures:
<pre> 
weights = [11, 33, 50, 0.4, 4]
</pre>
These emphasize the importance of:

Hapax legomena and lexical diversity
Average word length and sentence style with lesser emphasis
🚀 How to Use

1. Setup
Ensure Python 3 is installed. Place known texts in the known_authors/ folder and mystery texts in a suitable location.

2. Run Interactive Author Prediction
<pre> 
python authorship_identifier.py
</pre>
Follow prompts to input the path of the mystery file.

3. Run Non-Interactive Batch Test
   
This runs by default however, if you want only this to be displayed then
Comment the following call  
<pre> 
make_guess_interactive("known_authors") 
</pre>

✅ Running Tests

To verify all utilities are working correctly, run the test functions:
<pre> 
test_clean_word()
test_average_word_length()
test_different_to_total()
test_exactly_once_to_total()
test_split_string()
test_get_sentences()
test_average_sentence_length()
test_get_phrases()
test_average_sentence_complexity()
</pre>

📎 Dependencies

Pure Python (no external libraries needed except standard os, collections, and string modules)

📌 Notes

- Ensure UTF-8 encoding for all text files.
- Empty strings and whitespace-only content are safely handled.
- You can adjust the weights for better tuning depending on the dataset.
