"""Tokenization and character n-gram utilities."""
import re
import zlib

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text):
    """Lower-cased alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def char_ngrams(token, n_min=3, n_max=5):
    """Boundary-aware character n-grams of a token (the token is wrapped in '<' and '>')."""
    w = f"<{token}>"
    grams = []
    for n in range(n_min, n_max + 1):
        grams.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return grams or [w]


def ngram_ids(token, n_buckets, n_min=3, n_max=5):
    """Deterministic hashed ids of the character n-grams of a token (CRC32, independent of PYTHONHASHSEED)."""
    return [zlib.crc32(g.encode("utf8")) % n_buckets for g in char_ngrams(token, n_min, n_max)]
