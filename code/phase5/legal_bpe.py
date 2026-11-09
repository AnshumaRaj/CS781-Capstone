"""Byte-level BPE for CS781 Capstone Phase 1.

This is trained from scratch on the capstone corpus. It does not use
SentencePiece, Hugging Face tokenizers, or the GPT-2 vocabulary.

Assignment 1 merges the pair that occurs most often. On legal text that
fills the vocabulary with pieces of very common words such as "the" and
"of", even when those letters usually belong to other words.

Here a pair must occur at least `min_pair_count` times. Among those pairs
the one with the highest Dice-weighted count is merged:

    score(x, y) = C(xy) * 2 * C(xy) / (C(x) + C(y))

The fraction is near 1 when x and y almost always appear as this pair, and
near 0 when they usually appear apart. Multiplying by C(xy) keeps frequent
pairs ahead of rare ones. If two pairs have the same score, the higher count
wins. If the counts are also equal, the lexicographically smaller pair wins,
which is the Assignment 1 tie-break.

Merges stay inside a whitespace-separated word. A word-end marker is part of
that word, so a frequent whole word can become one token. Spaces stay
spaces, and uppercase stays uppercase. The first four vocabulary ids are
<pad>, <unk>, <s>, </s>. Every byte 0-255 has an id, so ordinary text does
not need <unk>.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter


END = 256  # word-end marker, not a UTF-8 byte
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SPECIALS = ("<pad>", "<unk>", "<s>", "</s>")
# Model ids 0-3 are the specials above. Internal symbol 0 (byte 0) is model id 4.
ID_OFFSET = 4
BASE_SYMBOLS = 257  # bytes 0-255 plus END
TOKEN_RE = re.compile(r"\s+|\S+")


def iter_texts(records):
    """Yield strings from a list, a dict of records, or a single record."""
    if isinstance(records, list):
        rows = records
    elif isinstance(records, dict):
        if isinstance(records.get("text"), str):
            rows = [records]
        else:
            rows = list(records.values())
    else:
        raise TypeError("records must be a list or a dict")
    for row in rows:
        if isinstance(row, str):
            yield row
        elif isinstance(row, dict) and isinstance(row.get("text"), str):
            yield row["text"]


class LegalBPE:
    def __init__(self):
        self.score_name = "dice"
        self.min_pair_count = 2
        self.requested_vocab_size = ID_OFFSET + BASE_SYMBOLS
        self.merges = []  # list of (left_internal, right_internal), in learn order
        self.symbol_bytes = {i: (i,) for i in range(256)}
        self.symbol_bytes[END] = (END,)
        self.merge_rank = {}
        self.merge_to = {}
        self.keys = {}
        self.history = []
        self._cache = {}
        self._rebuild_tables()

    # ------------------------------------------------------------------
    # Public attributes the Phase 1 notebook uses
    # ------------------------------------------------------------------

    @property
    def pad_token_id(self):
        return PAD_ID

    @property
    def unk_token_id(self):
        return UNK_ID

    @property
    def bos_token_id(self):
        return BOS_ID

    @property
    def eos_token_id(self):
        return EOS_ID

    pad_token = SPECIALS[0]
    unk_token = SPECIALS[1]
    bos_token = SPECIALS[2]
    eos_token = SPECIALS[3]

    def __len__(self):
        return ID_OFFSET + BASE_SYMBOLS + len(self.merges)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, texts, vocab_size=8192, min_pair_count=2, score="dice", max_chars=None):
        if score not in ("dice", "frequency"):
            raise ValueError("score must be 'dice' or 'frequency'")
        minimum = ID_OFFSET + BASE_SYMBOLS + 1
        if vocab_size < minimum:
            raise ValueError(f"vocab_size must be at least {minimum}")

        model = cls()
        model.score_name = score
        model.min_pair_count = int(min_pair_count)
        model.requested_vocab_size = int(vocab_size)
        target_merges = int(vocab_size) - (ID_OFFSET + BASE_SYMBOLS)

        word_freq = Counter()
        documents = 0
        seen_chars = 0
        for text in texts:
            documents += 1
            if not isinstance(text, str) or text == "":
                continue
            if max_chars is not None and seen_chars >= max_chars:
                break
            seen_chars += len(text)
            for tok in TOKEN_RE.findall(text):
                word_freq[model._token_symbols(tok)] += 1
            if documents % 100000 == 0:
                print(f"BPE counted {documents:,} documents, {len(word_freq):,} unique words")

        if not word_freq:
            raise ValueError("BPE training saw no text")

        print(
            f"BPE documents: {documents:,} | characters: {seen_chars:,} | "
            f"unique words: {len(word_freq):,} | score: {score} | target vocab: {vocab_size}"
        )
        model._learn(word_freq, target_merges)
        print(f"BPE finished. Vocabulary size: {len(model):,}. Merges: {len(model.merges):,}.")
        for left, right, count, value in model.history[:12]:
            print(f"  merge {left!r} + {right!r}  count={count}  score={value:.2f}")
        return model

    def _learn(self, word_freq, target_merges):
        self.word_freq = dict(word_freq)
        self.pair_counts = Counter()
        self.pair_words = {}
        self.sym_counts = Counter()
        self.sym_pairs = {}
        self.version = {}
        self.heap = []
        self.history = []
        self.merges = []
        self.merge_rank = {}
        self.merge_to = {}
        self._cache = {}
        self.symbol_bytes = {i: (i,) for i in range(256)}
        self.symbol_bytes[END] = (END,)
        self.keys = {i: self._key_for(i) for i in range(257)}

        dirty_pairs = set()
        dirty_syms = set()
        for word, freq in self.word_freq.items():
            self._register(word, freq, dirty_pairs, dirty_syms)
        self._rebuild_heap()

        next_id = 257
        while len(self.merges) < target_merges:
            if len(self.heap) > max(1000, 8 * len(self.pair_counts)):
                self._rebuild_heap()
            chosen = self._pop_best()
            if chosen is None:
                print(
                    f"BPE stopped early at vocab {len(self)}. "
                    f"No pair occurs at least {self.min_pair_count} times."
                )
                break
            left, right = chosen
            count = self.pair_counts[chosen]
            value = self._score(chosen)
            new_id = next_id
            next_id += 1
            self.symbol_bytes[new_id] = self.symbol_bytes[left] + self.symbol_bytes[right]
            self.keys[new_id] = self._key_for(new_id)
            self.merges.append((left, right))
            self.history.append((self._pretty(left), self._pretty(right), int(count), float(value)))

            dirty_pairs = set()
            dirty_syms = set()
            for word in list(self.pair_words.get(chosen, ())):
                freq = self.word_freq.pop(word, None)
                if freq is None:
                    continue
                self._unregister(word, freq, dirty_pairs, dirty_syms)
                merged = _merge_symbols(word, left, right, new_id)
                if merged in self.word_freq:
                    self.word_freq[merged] += freq
                    self._register(merged, freq, dirty_pairs, dirty_syms)
                else:
                    self.word_freq[merged] = freq
                    self._register(merged, freq, dirty_pairs, dirty_syms)

            affected = set(dirty_pairs)
            for sym in dirty_syms:
                affected.update(self.sym_pairs.get(sym, ()))
            affected.add(chosen)
            for pair in affected:
                self._push(pair)

            done = len(self.merges)
            if done <= 10 or done % 500 == 0:
                print(
                    f"  [{done}/{target_merges}] {self.history[-1][0]!r} + "
                    f"{self.history[-1][1]!r}  count={count}  score={value:.2f}"
                )

        self._rebuild_tables()
        # Training indexes are large. Drop them before save.
        self.word_freq = {}
        self.pair_counts = Counter()
        self.pair_words = {}
        self.sym_counts = Counter()
        self.sym_pairs = {}
        self.heap = []

    def _register(self, word, freq, dirty_pairs, dirty_syms):
        for sym in word:
            self.sym_counts[sym] += freq
            dirty_syms.add(sym)
        for pair, times in Counter(zip(word, word[1:])).items():
            self.pair_counts[pair] += freq * times
            self.pair_words.setdefault(pair, set()).add(word)
            for sym in pair:
                self.sym_pairs.setdefault(sym, set()).add(pair)
            dirty_pairs.add(pair)

    def _unregister(self, word, freq, dirty_pairs, dirty_syms):
        for sym in word:
            self.sym_counts[sym] -= freq
            if self.sym_counts[sym] <= 0:
                del self.sym_counts[sym]
            dirty_syms.add(sym)
        for pair, times in Counter(zip(word, word[1:])).items():
            self.pair_counts[pair] -= freq * times
            words = self.pair_words.get(pair)
            if words is not None:
                words.discard(word)
                if not words:
                    del self.pair_words[pair]
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
                for sym in pair:
                    bucket = self.sym_pairs.get(sym)
                    if bucket is not None:
                        bucket.discard(pair)
                        if not bucket:
                            del self.sym_pairs[sym]
                self.version[pair] = self.version.get(pair, 0) + 1
            else:
                dirty_pairs.add(pair)

    def _score(self, pair):
        count = self.pair_counts.get(pair, 0)
        if count < self.min_pair_count:
            return None
        if self.score_name == "frequency":
            return float(count)
        left = self.sym_counts.get(pair[0], 0)
        right = self.sym_counts.get(pair[1], 0)
        if left <= 0 or right <= 0:
            return None
        return (2.0 * count * count) / (left + right)

    def _push(self, pair):
        value = self._score(pair)
        if value is None:
            return
        version = self.version.get(pair, 0) + 1
        self.version[pair] = version
        count = self.pair_counts[pair]
        heapq.heappush(
            self.heap,
            (
                -value,
                -count,
                self.keys[pair[0]],
                self.keys[pair[1]],
                pair[0],
                pair[1],
                version,
            ),
        )

    def _rebuild_heap(self):
        self.heap = []
        for pair in list(self.pair_counts):
            self._push(pair)

    def _pop_best(self):
        while self.heap:
            _neg_value, _neg_count, _left_key, _right_key, left, right, version = heapq.heappop(self.heap)
            pair = (left, right)
            if self.version.get(pair) != version:
                continue
            if self._score(pair) is None:
                continue
            return pair
        return None

    def _rebuild_tables(self):
        self.merge_rank = {}
        self.merge_to = {}
        self.symbol_bytes = {i: (i,) for i in range(256)}
        self.symbol_bytes[END] = (END,)
        next_id = 257
        for rank, (left, right) in enumerate(self.merges):
            self.merge_rank[(left, right)] = rank
            self.merge_to[(left, right)] = next_id
            self.symbol_bytes[next_id] = self.symbol_bytes[left] + self.symbol_bytes[right]
            next_id += 1
        self.keys = {i: self._key_for(i) for i in self.symbol_bytes}
        self._cache = {}

    def _key_for(self, sym):
        parts = []
        for piece in self.symbol_bytes[sym]:
            parts.append("</w>" if piece == END else f"{piece:02x}")
        return "".join(parts)

    def _pretty(self, sym):
        raw = []
        end = False
        for piece in self.symbol_bytes[sym]:
            if piece == END:
                end = True
            else:
                raw.append(piece)
        try:
            text = bytes(raw).decode("utf-8")
            shown = text.encode("ascii", errors="backslashreplace").decode("ascii")
        except UnicodeDecodeError:
            shown = "".join(f"<{byte:02x}>" for byte in raw)
        return shown + "</w>" if end else shown

    def _token_symbols(self, token):
        raw = tuple(token.encode("utf-8"))
        if token.isspace():
            return raw
        return raw + (END,)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text, add_special_tokens=False):
        ids = []
        if add_special_tokens:
            ids.append(BOS_ID)
        if text:
            for tok in TOKEN_RE.findall(text):
                ids.extend(self._encode_token(tok))
        if add_special_tokens:
            ids.append(EOS_ID)
        return ids

    def encode_with_offsets(self, text):
        """Map text to token ids and the exact character span of each token.

        Merges never cross a whitespace boundary, and the word-end marker
        does not consume a character. A span is a half-open [start, end)
        into `text`, which is the same convention as the Phase 2 gold files.
        """
        ids = []
        spans = []
        for match in TOKEN_RE.finditer(text):
            token = match.group()
            origin = match.start()
            char_of_byte = []
            for index, char in enumerate(token):
                char_of_byte.extend([origin + index] * len(char.encode("utf-8")))
            byte_at = 0
            for model_id in self._encode_token(token):
                pieces = self.symbol_bytes.get(int(model_id) - ID_OFFSET, ())
                nbytes = sum(1 for piece in pieces if piece != END)
                if nbytes <= 0 or byte_at >= len(char_of_byte):
                    continue
                stop = min(byte_at + nbytes, len(char_of_byte))
                left = char_of_byte[byte_at]
                right = char_of_byte[stop - 1] + 1
                byte_at = stop
                if right > left:
                    ids.append(int(model_id))
                    spans.append((left, right))
        return ids, spans

    def _encode_token(self, token):
        cached = self._cache.get(token)
        if cached is None:
            symbols = list(self._token_symbols(token))
            while len(symbols) >= 2:
                best_rank = None
                best_index = None
                for i in range(len(symbols) - 1):
                    rank = self.merge_rank.get((symbols[i], symbols[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_index = i
                if best_index is None:
                    break
                pair = (symbols[best_index], symbols[best_index + 1])
                symbols = symbols[:best_index] + [self.merge_to[pair]] + symbols[best_index + 2 :]
            cached = tuple(sym + ID_OFFSET for sym in symbols)
            self._cache[token] = cached
        return cached

    def decode(self, ids, skip_special_tokens=True):
        raw = []
        for model_id in ids:
            model_id = int(model_id)
            if model_id < ID_OFFSET:
                continue
            internal = model_id - ID_OFFSET
            pieces = self.symbol_bytes.get(internal)
            if pieces is None:
                continue
            for piece in pieces:
                if piece != END:
                    raw.append(piece)
        return bytes(raw).decode("utf-8", errors="replace")

    def batch_encode(self, texts, max_length=None, padding="left"):
        """Pad a batch for generation. Keep the end of a long prompt."""
        rows = []
        for text in texts:
            ids = self.encode(text, add_special_tokens=False)
            if max_length is not None and len(ids) > max_length:
                ids = ids[-max_length:]
            rows.append(ids)
        width = max((len(row) for row in rows), default=0)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad = width - len(row)
            if padding == "left":
                input_ids.append([PAD_ID] * pad + row)
                attention_mask.append([0] * pad + [1] * len(row))
            else:
                input_ids.append(row + [PAD_ID] * pad)
                attention_mask.append([1] * len(row) + [0] * pad)
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        payload = {
            "format": "cs781-legal-bpe-v1",
            "score": self.score_name,
            "min_pair_count": self.min_pair_count,
            "vocab_size": len(self),
            "specials": list(SPECIALS),
            "merges": self.merges,
            "preview": [f"{left} + {right}" for left, right, _count, _value in self.history[:40]],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        print("Saved BPE to", path)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "cs781-legal-bpe-v1":
            raise ValueError(f"{path} is not a cs781-legal-bpe-v1 file")
        model = cls()
        model.score_name = payload.get("score", "dice")
        model.min_pair_count = int(payload.get("min_pair_count", 2))
        model.merges = [(int(left), int(right)) for left, right in payload["merges"]]
        model.requested_vocab_size = int(payload.get("vocab_size", len(model)))
        model.history = []
        model._rebuild_tables()
        return model


def _merge_symbols(word, left, right, new_id):
    merged = []
    index = 0
    size = len(word)
    while index < size:
        if index + 1 < size and word[index] == left and word[index + 1] == right:
            merged.append(new_id)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def _check():
    sticky = ["ab"] * 6
    loose = ["mn", "mp", "mq", "mr", "ms"] * 10
    one_merge = ID_OFFSET + BASE_SYMBOLS + 1

    dice = LegalBPE.train(sticky + loose, vocab_size=one_merge, min_pair_count=2, score="dice")
    freq = LegalBPE.train(sticky + loose, vocab_size=one_merge, min_pair_count=2, score="frequency")
    assert dice.history[0][0] == "a" and dice.history[0][1] == "b", dice.history[0]
    assert freq.history[0][0] in list("mnmpqrms") or freq.history[0][0].startswith("m"), freq.history[0]
    assert not (freq.history[0][0] == "a" and freq.history[0][1] == "b"), freq.history[0]

    samples = [
        "The Supreme Court held that the petitioner is entitled to relief.",
        "Section 302 of the Indian Penal Code\nwas applied.",
        "court Court COURT",
        "न्यायालय ने आदेश दिया।",
        "  spaced   words\tand tabs",
    ]
    full = LegalBPE.train(samples * 30, vocab_size=400, min_pair_count=2, score="dice")
    for text in samples:
        assert full.decode(full.encode(text)) == text, text
    assert full.encode("Court") != full.encode("court")
    joined = full.decode(full.encode("ab cd"))
    assert joined == "ab cd"
    # A space is its own token, so a merge cannot glue the b in "ab" to the c in "cd".
    encoded = full.encode("ab cd")
    assert len(encoded) >= 2

    path = "_legal_bpe_selfcheck.json"
    full.save(path)
    loaded = LegalBPE.load(path)
    for text in samples:
        assert loaded.decode(loaded.encode(text)) == text
    assert len(loaded) == len(full)
    ids, mask = loaded.batch_encode(["short", "a longer prompt"], max_length=8, padding="left")
    assert len(ids[0]) == len(ids[1])
    assert mask[0][0] == 0 or len(loaded.encode("short")) == len(ids[0])
    import os
    os.remove(path)
    print("legal_bpe self-check passed")


if __name__ == "__main__":
    _check()
