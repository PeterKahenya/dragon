from collections import defaultdict, Counter
from typing import Any
import regex as re
import multiprocessing as mp
import seaborn as sns

# count words
def count_words(doc: Any) -> dict:
    """
        - Split text into segments using regex 
        - Encode the segments into tuple of utf-8 codepoints
        - Count the frequency of each segment
    """
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
    pattern = re.compile(pattern)
    segments = re.findall(pattern, doc)
    segments = [tuple(segment.encode('utf-8')) for segment in segments]
    return Counter(segments)

# count words in a list of documents
def count_words_in_documents(documents: list, processes: int = 1) -> dict:
    """
        - Count the frequency of each segment in a list of documents
    """
    chunk_size = (len(documents) // processes) if len(documents) > processes else 1
    print(chunk_size)
    with mp.Pool(processes = processes) as pool:
        word_counts = pool.map(count_words, documents, chunksize=chunk_size)
    word_counts = sum(word_counts, Counter())
    return word_counts

# count pairs of tokens
def count_pairs(word_count: tuple) -> dict:
    """
        - Count the frequency of each pair of tokens
    """
    pairs = defaultdict(int)
    word, count = word_count
    for p1, p2 in zip(word[:-1], word[1:]):
        pairs[(p1, p2)] += count
    return pairs

# count pairs of tokens in a list word counts
def count_pairs_in_word_counts(word_counts: dict, processes: int = 1) -> dict:
    """
        - Count the frequency of each pair of tokens in a list of word counts
    """
    chunk_size = len(word_counts) // processes
    word_counts = list(word_counts.items())
    pairs = defaultdict(int)
    with mp.Pool(processes) as pool:
        results = pool.map(count_pairs, word_counts, chunksize=chunk_size)
    for res in results:
        for k,v in res.items():
            pairs[k] += v
    return pairs

# merge_pairs in one word count
def merge_pairs(word_count_arg: tuple) -> dict:
    """
        - Merge pairs of tokens in a word count
    """
    word_count, pair, new_tok = word_count_arg
    toks, freq = list(word_count.items())[0]
    new_tokens = []
    i = 0
    while i < len(toks):
        if i < len(toks) - 1 and (toks[i], toks[i + 1]) == pair:
            new_tokens.append(new_tok)
            i += 2
        else:
            new_tokens.append(toks[i])
            i += 1
    return {tuple(new_tokens): freq}

# merge_pairs_in_word_counts in a list of word counts
def merge_pairs_in_word_counts(word_counts: dict, pair: tuple[int, int], new_tok: int, processes: int = 1) -> dict:
    """
        - Merge pairs of tokens in a list of word counts
    """
    chunk_size = len(word_counts) // processes
    word_counts_arg = [({t:f}, pair, new_tok) for t,f in word_counts.items()]
    with mp.Pool(processes) as pool:
        results = pool.map(merge_pairs, word_counts_arg, chunksize=chunk_size)
    merged_word_counts = defaultdict(int)
    for res in results:
        for k,v in res.items():
            merged_word_counts[k] += v
    return merged_word_counts


# @lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    credit - openai/gpt2
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
            
def plot_loss_curve(epochs,train_loss_values,test_loss_values):
    sns.lineplot(x=epochs,y=train_loss_values)
    if test_loss_values:
        sns.lineplot(x=epochs,y=test_loss_values)
