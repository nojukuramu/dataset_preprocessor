## DATASET PREPROCESSOR
Python Script for preprocessing Tagalog Datasets.

# Features
1. Cleans text data by removing html tags and symbols.
2. Remove other languages or sentences with mixed language (Tagalog is default but can be modified to other languages.)
3. Batch size is modifiable depending on memory. (Parallel Processing not available yet. 1024 bytes default)
4. Checkpoints for FastText version to handle large datasets.

Note: This project is defaulted to tagalog but can be modified on different language available to langdetect library.

# To be implemented
- config file for easy modification.
