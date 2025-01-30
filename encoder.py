import regex as re
import utils
from tqdm import tqdm
import json
import os
import numpy as np
import multiprocessing as mp


class TextEncoder:
    """
        Byte-pair encoding tokenizer.
    """
    def __init__(self):
        self.merges = {}
        self.vocab = self._build_vocab()
        self.special_tokens_list = ['<|endofdoc|>']    
        self.special_tokens = {}
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        self.pattern_compiled = re.compile(self.pattern)
        self.version = 0
        
    def _add_special_tokens_to_vocab(self):
        main_vocab_length = len(self.vocab)
        for i, token in enumerate(self.special_tokens_list):
            idx = main_vocab_length+i
            self.special_tokens[token] = idx
            self.vocab[idx] = token

    def _build_vocab(self):
        vocab = utils.bytes_to_unicode()
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    
    def train(self, docs: list[str], vocab_size: int = 256, processes: int = 1):
        """
            Builds the merges for the tokenizer using the given list of wordcounts.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        word_counts = utils.count_words_in_documents(docs, processes=processes)
        num_merges = vocab_size - 256
        initial_vocab_size = len(self.vocab)
        for i in tqdm(range(num_merges)):
            pairs = utils.count_pairs_in_word_counts(word_counts, processes)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_tok = initial_vocab_size + i
            word_counts = utils.merge_pairs_in_word_counts(word_counts, best_pair, new_tok, processes)
            self.merges[best_pair] = new_tok
            self.vocab[new_tok] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # add the new token to the vocab assign it the value of the two merged tokens' binary values
        self.version += 1
        self._add_special_tokens_to_vocab()
        
    def _merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        """
        Merges a pair of tokens in a list of tokens.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
        
    def encode(self, text: str) -> list[int]:
        """
        Given a string, encodes it into a list of token IDs.
        """
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text) # returns a list of text separated by the special tokens with the tokens themselves in the list as well
        tokens = []
        for chunk in special_chunks:
            if chunk in self.special_tokens:
                tokens.append(self.special_tokens[chunk])
            else:
                text_chunks = re.findall(self.pattern_compiled, chunk)
                for part in text_chunks:
                    part_tokens = list(part.encode("utf-8")) # encode the part into bytes
                    while len(part_tokens) >= 2:
                        pairs = [pair for pair in zip(part_tokens, part_tokens[1:])]
                        pair = min(pairs, key=lambda p: self.merges.get(p,float("inf")))
                        if pair not in self.merges:
                            break
                        part_tokens = self._merge_tokens(part_tokens, pair, self.merges[pair])
                    tokens.extend(part_tokens)
        return tokens
    
    def decode(self, tokens: list[int]) -> str:
        """
        Given a list of token IDs, decodes them into a string.
        """
        inverse_special_tokens = {idx: st for st, idx in self.special_tokens.items()}
        inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        part_strs = []
        for idx in tokens:
            if idx in self.vocab:
                part_strs.append(self.vocab[idx])
            elif idx in inverse_special_tokens:
                part_strs.append(inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text = "".join(part_strs)
        text = bytearray([inverse_vocab[c] for c in text]).decode('utf-8', errors="replace")
        return text
    
    def save(self, path: str):
        """
            Saves the tokenizer to a file. Saves the merges, special tokens and pattern.
        """
        os.makedirs(path, exist_ok=True)
        merge_file, vocab_file = os.path.join(path, "tokenizer.merges"), os.path.join(path, "tokenizer.json")
        with open(merge_file, 'w', encoding='utf-8') as f:
            f.write(f"#dragontokenizer: {self.version}\n")
            f.write(f"#pattern: {self.pattern}\n")
            f.write(f"#special_tokens_count: {len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        vocab = {}
        for idx, token in self.vocab.items():
            if idx in inverted_merges:
                idx0, idx1 = inverted_merges[idx]
                s0 = self.vocab[idx0]
                s1 = self.vocab[idx1]
                vocab[s0+s1] = idx
            else:
                vocab[token] = idx
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f)
        
    @classmethod
    def load(cls, path: str):
        """Inverse of save() but only for the merges file"""
        model_file = os.path.join(path, "tokenizer.merges")
        merges, special_tokens, idx = {}, {}, 256
        tokenizer = cls()
        with open(model_file, 'r', encoding="utf-8") as f:
            version = int(f.readline().split(":")[1].strip())
            pattern = f.readline().strip()
            num_special = int(f.readline().split(":")[1].strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        tokenizer.version = version
        tokenizer.pattern = pattern
        tokenizer.merges = merges
        tokenizer.special_tokens = special_tokens
        tokenizer.vocab = tokenizer._build_vocab()
        tokenizer._add_special_tokens_to_vocab()
        return tokenizer
    
# tokenize a document
def tokenize(args: tuple[str, str]) -> np.ndarray:
    doc, enc_path = args
    enc = TextEncoder.load(path=enc_path)
    eot = enc.special_tokens['<|endofdoc|>']
    tokens = [eot]
    tokens.extend(enc.encode(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
def tokenize_and_write_shards(ds: list, shards_dir: str, shard_size: int = 1000, nprocs: int = 1, model_path: str = "models/124M") -> int:
    os.makedirs(shards_dir, exist_ok=True)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        current_shard = []
        current_size = 0
        ds_and_enc = [(doc, model_path) for doc in ds]
        total_tokens = 0
        for tokens in tqdm(pool.imap(tokenize, ds_and_enc, chunksize=1)):
            total_tokens += len(tokens)
            for token in tokens:
                current_shard.append(token)
                current_size += 1
                if current_size >= shard_size:
                    split = "val" if shard_index == 0 else "train"
                    shard_path = os.path.join(shards_dir, f"{split}_{shard_index:06d}.npy")
                    np.save(shard_path, current_shard)
                    print(f"Saved {shard_path}")
                    current_shard = []
                    current_size = 0
                    shard_index += 1
        if current_shard:
            split = "val" if shard_index == 0 else "train"
            shard_path = os.path.join(shards_dir, f"{split}_{shard_index:06d}.npy")
            np.save(shard_path, current_shard)
            print(f"Saved {shard_path}")
            
    return total_tokens