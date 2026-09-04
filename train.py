
import regex
import re
from collections import Counter

# initializing vocabulary

vocab = {bytes([i]): i for i in range(256)}
# regex pattern for spliting
PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
)

def detect_end_of_text(text: str, special_token_id: int):
    parts = re.split(r"(<\|endoftext\|>)", text)
    parts = [part if part != "<|endoftext|>" else special_token_id for part in parts]

    return parts


def pretokenize(
    text: str,
    special_token_id: int
):
    PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    parts = detect_end_of_text(text, special_token_id)

    token_sequences = Counter()

    for part in parts:

        if part == "":
            continue

        # Special token
        if isinstance(part, int):
            token_sequences[(part,)] += 1
            continue

        # Normal text
        chunks = PATTERN.findall(part)

        for chunk in chunks:

            byte_sequence = tuple(
                bytes([byte])
                for byte in chunk.encode("utf-8")
            )

            token_sequences[byte_sequence] += 1

    return dict(token_sequences)



def pair_counts(text):

    pair_counts = {}

    for sequence, frequency in text.items():

        for i in range(len(sequence) - 1):

            pair = (sequence[i], sequence[i + 1])

            pair_counts[pair] = (
                pair_counts.get(pair, 0) + frequency
            )

    if not pair_counts:
        return None

    best_pair = max(
        pair_counts,
        key=lambda pair: (pair_counts[pair], pair)
    )

    return best_pair, pair_counts[best_pair]


def merge_sequence(text, pair):
    

    new_pretokenized = {}

    for sequence, frequency in text.items():

        new_sequence = []
        i = 0

        while i < len(sequence):

            # Le pair correspond
            if (
                i < len(sequence) - 1
                and sequence[i] == pair[0]
                and sequence[i + 1] == pair[1]
            ):

                merged_token = sequence[i] + sequence[i + 1]

                new_sequence.append(merged_token)

                i += 2

            else:

                new_sequence.append(sequence[i])

                i += 1

        new_sequence = tuple(new_sequence)

        # Additionner les fréquences si la séquence existe déjà
        new_pretokenized[new_sequence] = (
            new_pretokenized.get(new_sequence, 0)
            + frequency
        )

    return new_pretokenized


def bpeAlgorithme(
        text: str,
        vocabulary: dict[bytes, int],
        special_token_id: int,
        vocab_size: int
):

    # 1. Pretokenization
    important_dict = pretokenize(
        text,
        special_token_id
    )

    # Liste des merges dans l'ordre où ils ont été appris
    merges = []

    # 2. Continuer jusqu'à atteindre vocab_size
    while len(vocabulary) < vocab_size:

        result = pair_counts(important_dict)

        # Plus aucune paire disponible
        if result is None:
            break

        best_pair, frequency = result

        # 3. Créer le nouveau token
        merged_token = best_pair[0] + best_pair[1]

        # ID du nouveau token
        new_id = max(vocabulary.values()) + 1

        vocabulary[merged_token] = new_id

        # 4. Sauvegarder le merge
        merges.append(best_pair)

        # 5. Appliquer le merge au corpus
        important_dict = merge_sequence(
            important_dict,
            best_pair
        )

    return vocabulary, merges

