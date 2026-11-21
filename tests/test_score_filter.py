"""Temporal score filter (score_filter_decay) through the data loader.

The fixture games give every position the same side-to-move score S. Following
positions are seen from the other side every second ply, so a position with m
usable following positions (m <= 10) gets the filtered score

    s_m = 380 * logit(sum_k d^k q_k / sum_k d^k),  k = 0..m,
    q_k = sigmoid(S / 380) for even k, 1 - sigmoid(S / 380) for odd k,

and every delivered score must be one of s_0 (= S) .. s_10, with s_10 by far
the most common one (games are 160 plies long).
"""

import math
import os
import subprocess
import sys
import threading
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_dataloader_read_balance import (
    REPO_ROOT,
    SKIP_ALL_OFF,
    _build_generator,
    _fixture_dir,
)

from data_loader import stream
from data_loader.config import (
    DataloaderDDPConfig,
    DataloaderIOConfig,
    DataloaderSkipConfig,
)

SCALE = 380.0
MAX_PLIES = 10
POSITIONS_PER_FILE = 20000

# Where scripts/compile_data_loader.sh and the CMake build put the library.
LOADER_BUILT = any(
    os.path.exists(os.path.join(REPO_ROOT, *d, "libtraining_data_loader.so"))
    for d in (("build",), ("data_loader", "cpp", "build"))
)


def _expected_scores(score, decay):
    """Filtered score for 0..MAX_PLIES usable following positions."""
    p = 1.0 / (1.0 + math.exp(-score / SCALE))
    out = []
    for m in range(MAX_PLIES + 1):
        num = sum(decay**k * (p if k % 2 == 0 else 1.0 - p) for k in range(m + 1))
        den = sum(decay**k for k in range(m + 1))
        q = num / den
        out.append(round(SCALE * math.log(q / (1.0 - q))))
    return out


def _skip_config(decay, **kwargs):
    fields = dict(SKIP_ALL_OFF.__dict__)
    fields.update(kwargs, score_filter_decay=decay)
    return DataloaderSkipConfig(**fields)


def _constant_score_files(score):
    generator = _build_generator()
    files = []
    for i in range(2):
        path = os.path.join(_fixture_dir(), f"constant_{score}_{i}.binpack")
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            if os.path.exists(path):
                os.remove(path)  # writer appends to existing files
            subprocess.run(
                [generator, path, str(POSITIONS_PER_FILE), str(300 + i), str(score)],
                check=True,
            )
        files.append(path)
    return files


def _all_scores(files, config):
    """All scores of one non-cyclic pass over the files."""
    st = stream.create_sparse_batch_stream(
        b"K32H3",
        4,
        files,
        4096,
        False,
        config,
        DataloaderDDPConfig(0, 1),
        None,
        DataloaderIOConfig(balance_window_mb=32),
    )
    result = {}

    def consume():
        scores = []
        while True:
            b = stream.fetch_next_sparse_batch(st)
            if not b:
                break
            n = b.contents.size
            scores.append(np.ctypeslib.as_array(b.contents.score, shape=(n,)).copy())
            stream.destroy_sparse_batch(b)
        result["scores"] = np.concatenate(scores) if scores else np.array([])

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(timeout=60.0)
    stream.destroy_sparse_batch_stream(st)
    if t.is_alive():
        raise AssertionError("non-cyclic stream did not terminate")
    return np.rint(result["scores"]).astype(int)


@unittest.skipUnless(LOADER_BUILT, "libtraining_data_loader.so not built")
class TestScoreFilter(unittest.TestCase):
    SCORE = 150

    @classmethod
    def setUpClass(cls):
        cls.files = _constant_score_files(cls.SCORE)

    def assert_filtered(self, scores, decay):
        expected = _expected_scores(self.SCORE, decay)
        # +-1 for rounding differences between C++ and Python.
        allowed = {s + d for s in expected for d in (-1, 0, 1)}
        unexpected = sorted(set(scores.tolist()) - allowed)
        self.assertEqual(unexpected, [], f"scores outside {expected}")
        most_common, _ = Counter(scores.tolist()).most_common(1)[0]
        self.assertLessEqual(abs(most_common - expected[MAX_PLIES]), 1)

    def test_disabled_keeps_scores(self):
        scores = _all_scores(self.files, _skip_config(0.0))
        self.assertEqual(len(scores), 2 * POSITIONS_PER_FILE)
        self.assertTrue((scores == self.SCORE).all())

    def test_filtered_scores(self):
        for decay in (0.3, 0.7):
            with self.subTest(decay=decay):
                scores = _all_scores(self.files, _skip_config(decay))
                # The filter changes scores, never which positions are kept.
                self.assertEqual(len(scores), 2 * POSITIONS_PER_FILE)
                self.assertLess(scores.mean(), self.SCORE)
                self.assert_filtered(scores, decay)

    def test_random_skips_do_not_shorten_the_average(self):
        """Positions dropped by random skipping still count as neighbours."""
        scores = _all_scores(self.files, _skip_config(0.3, random_fen_skipping=3))
        self.assertGreater(len(scores), 0)
        self.assertLess(len(scores), POSITIONS_PER_FILE)
        self.assert_filtered(scores, 0.3)


@unittest.skipUnless(LOADER_BUILT, "libtraining_data_loader.so not built")
class TestScoreFilterMateScores(unittest.TestCase):
    """Mate / tablebase range scores are not averaged and cut the chain."""

    def test_mate_scores_unchanged(self):
        score = 31800
        files = _constant_score_files(score)
        scores = _all_scores(files, _skip_config(0.3))
        self.assertEqual(len(scores), 2 * POSITIONS_PER_FILE)
        self.assertTrue((scores == score).all())


if __name__ == "__main__":
    unittest.main()
