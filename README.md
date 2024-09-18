## DATASET PREPROCESSOR
Python Script for preprocessing Tagalog Datasets. (modifiable for all language)

# 2 Versions
1. Lang Library version - Used for smaller datasets (no longer be updated)
2. FastText Version - Used for Relatively Large Datasets

# Features
1. Cleans text data by removing html tags and symbols.
2. Remove other languages or sentences with mixed language (Tagalog is default but can be modified to other languages.)
3. Batch size is modifiable depending on memory. (Parallel Processing not available yet. 1024 bytes default)
4. Checkpoints for FastText version to handle large datasets.

Note: This project is defaulted to tagalog but can be modified on different language available to langdetect library.

# To be implemented
- Config file for easy modification.
- Parallel Batch Computing
