# BPE Tokenizer (CS336 — Assignment 1)

A from-scratch implementation of a **byte-level Byte Pair Encoding (BPE) tokenizer**, written for
Stanford **CS336: Language Modeling from Scratch**.

This is a **first working version**: the full train → encode → decode loop runs end to end and
round-trips text correctly, but it has not yet been run against the official assignment test suite
and no performance work has been done. It is intentionally kept simple and readable so it can be
improved step by step (see [Roadmap](#roadmap)).

```
Tokenized text: [337, 346, 362, 334, 402, 357, 334, 308, 404, 307, 571, 404, 262, 105, 116, 104, 32, 80, 121, 116, 104, 376, 46]
Decoded text:   The intelligent student is learning machine learning with Python.
```

---

## How BPE works here

1. **Pre-tokenization** — the text is split with a GPT-2 style regex so that merges can never cross
   word boundaries. `<|endoftext|>` is pulled out first and replaced by a special token id.
2. **Byte encoding** — every pre-token is turned into a tuple of single bytes (`b'h'`, `b'e'`, ...),
   so the tokenizer works on *any* UTF-8 input, including accented and non-Latin text.
3. **Training** — repeatedly count all adjacent byte-pairs, merge the most frequent one into a new
   token, and record the merge, until the vocabulary reaches `vocab_size` (or no pair is left).
4. **Encoding** — re-apply the learned merges to new text, always choosing the merge with the
   **lowest vocabulary id** first, which reproduces the order the merges were learned in.
5. **Decoding** — map ids back to byte strings, concatenate, and decode as UTF-8.

---

## Project structure

```
bpe-tokenizer/
├── bpeTokenizer.py     # entry point / demo: trains on a small corpus, then encodes and decodes
├── train.py            # pre-tokenization + the BPE training loop
├── encode.py           # applies learned merges to new text and maps tokens to ids
├── decode.py           # maps ids back to text
├── requirements.txt    # Python dependencies
└── README.md
```

### `train.py` — building the vocabulary

| Name | Kind | Role |
|---|---|---|
| `vocab` | dict `bytes -> int` | Initial vocabulary: the 256 single bytes mapped to ids `0..255`. |
| `PATTERN` | compiled regex | GPT-2 style pre-tokenization pattern (contractions, words, numbers, punctuation, whitespace runs). |
| `detect_end_of_text(text, special_token_id)` | function | Splits the text on `<\|endoftext\|>` and replaces each occurrence with the integer `special_token_id`, returning a mixed list of strings and ints. |
| `pretokenize(text, special_token_id)` | function | Turns the corpus into a **frequency table**: `{tuple_of_bytes: count}`. Special tokens are stored as `(special_token_id,)`. Counting identical pre-tokens once is what keeps training tractable. |
| `pair_counts(text)` | function | Counts every adjacent pair across the frequency table and returns `(best_pair, count)`. Ties are broken by the greater pair (lexicographic on the byte strings). Returns `None` when no pair is left. |
| `merge_sequence(text, pair)` | function | Rewrites the whole frequency table with `pair` merged into a single token, summing frequencies of sequences that collapse onto each other. |
| `bpeAlgorithme(text, vocabulary, special_token_id, vocab_size)` | function | The training loop. Pre-tokenizes, then repeatedly picks the best pair, adds the merged token to `vocabulary` with a fresh id, appends the pair to `merges`, and applies the merge. Returns `(vocabulary, merges)`. |

### `encode.py` — text → token ids

| Name | Kind | Role |
|---|---|---|
| `pretokenize_encode(text, special_token_id)` | function | Same splitting as training, but returns an **ordered list** of byte tuples instead of a frequency table — order matters when encoding. |
| `apply_merge(sequence, merge_pair)` | function | Single left-to-right pass over one pre-token, replacing every occurrence of `merge_pair` with the concatenated token. |
| `verify_merge_dispo(sequence, merges)` | function | Lists which learned merges are currently applicable inside a sequence. |
| `merging_match(text, merges, vocab)` | function | Applies merges until none are left, each round picking the applicable pair whose merged token has the **smallest vocab id** — i.e. the earliest-learned merge wins. |
| `encode(text, merges, vocab, special_token_id)` | function | Public API: returns the flat `list[int]` of token ids. |
| `test`, `vocab`, `merges`, `new_dictionary` | module-level | Hard-coded toy vocabulary and merge list (a small `hello / yellow / fella` example) used while developing `merging_match`. They run on import and are not used by the public API. |

### `decode.py` — token ids → text

| Name | Kind | Role |
|---|---|---|
| `decode(tokens_list, vocab, special_token_id)` | function | Inverts `vocab` into `id -> bytes`, looks up every id, joins the byte strings and decodes them as UTF-8. |

### `bpeTokenizer.py` — the demo

Builds the 256-byte starting `vocab`, trains on an inline `corpus` (repeated words, numbers,
punctuation, French/Unicode text and two `<|endoftext|>` boundaries), then encodes and decodes a
sample sentence to show the round-trip.

---

## Running it

```bash
pip install -r requirements.txt
python bpeTokenizer.py
```

Requires Python 3.9+ (the code uses `dict[bytes, int]` style annotations).

---


## Notes

Written as coursework for Stanford CS336. The implementation, including the training loop and the
merge-priority logic, is my own.
