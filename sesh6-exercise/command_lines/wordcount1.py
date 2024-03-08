from wordcount import read_file, word_count, print_counts

data = read_file("sample-text.txt")      # add the filename
counts = word_count(data) # add appropriate function arguments
print_counts(counts)         # add appropriate function arguments