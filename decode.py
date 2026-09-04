
def decode(
        tokens_list:list,
        vocab: dict[bytes, int],
        special_token_id: int = 1500
        ):
    id_to_token = {id: token for token, id in vocab.items()}
    text = [id_to_token[token] for token in tokens_list]
    text = b''.join(text).decode("utf-8")
    return text

    