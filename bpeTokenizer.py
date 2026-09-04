import train
import encode
import decode

vocab = {bytes([i]): i for i in range(256)}

corpus = """
Hello, my boy! Hello, my friend.
This is a simple corpus for training a BPE tokenizer.
The tokenizer should learn frequent patterns, repeated words, and common sequences.

Hello hello hello world world world.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog again.

Machine learning is fascinating. Deep learning is powerful.
Artificial intelligence is changing the world.
Data science combines mathematics, statistics, programming, and domain knowledge.

I don't know if you're ready, but we're going to test the tokenizer.
It's important to understand how a tokenizer works.
I've written this tokenizer myself, and I'll improve it step by step.

Numbers are important too: 1, 10, 100, 1000, 2026, 3.14, 42.
Punctuation matters: ! ? . , ; : - _ ( ) [ ] { } / \\ @ # $ % & * + =

Some words repeat repeatedly because repeated patterns are what BPE learns.
token tokenizer tokens tokenize training train trained
hello hello hello hello
world world world world
computer computer computer computer
programming programming programming programming

Unicode should work too:
Algeria, Béjaïa, français, étudiant, ingénieur, intelligence artificielle.
Whitespace    should   also
be
tested.

Finally, we include the special token boundary.
<|endoftext|>
This is a new document.
<|endoftext|>
Another document starts here.
"""

result_vocab, result_merges = train.bpeAlgorithme(
    corpus,
    vocab,
    special_token_id=1000,
    vocab_size=3000
)

text_to_tokenize = "The intelligent student is learning machine learning with Python."

tokenized_text = encode.encode(
    text_to_tokenize,
    result_merges,
    result_vocab,
    special_token_id=1000
)

print("Tokenized text:", tokenized_text)

decoded_text = decode.decode(
    tokenized_text,
    result_vocab,
    special_token_id=1000
)

print("Decoded text:", decoded_text)