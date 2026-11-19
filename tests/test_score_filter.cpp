// Standalone unit test for the temporal score filter.
// Build & run:
//   g++ -std=c++20 -O1 -I data_loader/cpp/lib tests/test_score_filter.cpp -o /tmp/test_score_filter && /tmp/test_score_filter

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "score_filter.h"

using binpack::TrainingDataEntry;

static int failures = 0;

static const char* kStartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// Black to move and in check (Qh5-e8 after 1. e4 f5 2. Qh5+).
static const char* kCheckFen = "rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2";
// White to move, e4xd5 is a capture.
static const char* kCaptureFen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";

// Entries on the start position with a quiet move (e2-e4), consecutive plies
// starting at 10 unless given, and decisive alternating results unless given.
static std::vector<TrainingDataEntry> makeEntries(const std::vector<int>& scores,
                                                  const std::vector<int>& plies = {},
                                                  const std::vector<int>& results = {})
{
    std::vector<TrainingDataEntry> entries(scores.size());
    const auto start = chess::Position::fromFen(kStartFen);
    for (std::size_t i = 0; i < scores.size(); ++i)
    {
        entries[i].pos = start;
        entries[i].move = chess::Move{chess::e2, chess::e4};
        entries[i].score = static_cast<std::int16_t>(scores[i]);
        entries[i].ply = static_cast<std::uint16_t>(i < plies.size() ? plies[i] : 10 + i);
        entries[i].result = static_cast<std::int16_t>(i < results.size() ? results[i] : (i % 2 ? -1 : 1));
    }
    return entries;
}

static void checkEntries(const char* name,
                         const std::vector<TrainingDataEntry>& entries,
                         const std::vector<char>& cont,
                         score_filter::Config cfg,
                         const std::vector<int>& expected,
                         const std::vector<char>& needed = {})
{
    std::vector<std::int16_t> out;
    score_filter::apply(entries, cont, cfg, out, needed.empty() ? nullptr : &needed);

    bool ok = out.size() == expected.size();
    for (std::size_t i = 0; ok && i < expected.size(); ++i)
        ok = out[i] == expected[i];

    std::printf("%s %s", ok ? "ok  " : "FAIL", name);
    if (!ok)
    {
        ++failures;
        std::printf("   got:");
        for (auto v : out) std::printf(" %d", int(v));
        std::printf("   expected:");
        for (auto v : expected) std::printf(" %d", v);
    }
    std::printf("\n");
}

static void check(const char* name,
                  const std::vector<int>& scores,
                  const std::vector<char>& cont,
                  double decay,
                  const std::vector<int>& expected,
                  const std::vector<int>& plies = {})
{
    checkEntries(name, makeEntries(scores, plies), cont, {decay, true}, expected);
}

// Expected filtered score from an explicit list of contributing positions
// {k plies ahead, raw score}, straight from the formula in score_filter.h.
static int expect(double d, const std::vector<std::pair<int, int>>& terms)
{
    double sum = 0, weight = 0;
    for (auto [k, s] : terms)
    {
        const double w = std::pow(d, k);
        const double q = 1.0 / (1.0 + std::exp(-s / 380.0));
        sum += w * (k % 2 == 0 ? q : 1.0 - q);
        weight += w;
    }
    const double p = std::clamp(sum / weight, 1e-9, 1.0 - 1e-9);
    return static_cast<int>(std::lround(380.0 * std::log(p / (1.0 - p))));
}

int main()
{
    const double d = 0.5;

    // Plain chain (side-to-move relative scores, alternating perspective).
    // Hand check of entry 1 -> 0 below: 380*logit((0.5654 + 0.5*0.5524)/1.5) = 93.
    check("chain", {100, -80, 60}, {0, 1, 1}, d,
          {expect(d, {{0, 100}, {1, -80}, {2, 60}}), expect(d, {{0, -80}, {1, 60}}), 60});
    check("chain break", {100, -80, 60}, {0, 1, 0}, d, {93, -80, 60});

    // Mate / VALUE_NONE / tablebase cut the chain and stay untouched; the entry
    // before has no usable neighbour and keeps its raw score.
    check("mate cut", {100, 32000, 60}, {0, 1, 1}, d, {100, 32000, 60});
    check("value none cut", {100, 32002, 60}, {0, 1, 1}, d, {100, 32002, 60});
    check("tb cut", {100, -31800, 60}, {0, 1, 1}, d, {100, -31800, 60});

    // Decay 0: identity.
    check("decay 0", {100, -80, 60}, {0, 1, 1}, 0.0, {100, -80, 60});

    // Constant evaluation along a game: no-op.
    check("constant eval", {50, -50, 50, -50, 50}, {0, 1, 1, 1, 1}, 0.9, {50, -50, 50, -50, 50});

    // Two games back to back (isCont[2] = 0 separates them).
    check("two games", {100, -100, 40, -40}, {0, 1, 0, 1}, d, {100, -100, 40, -40});

    // Search-tree sample (ply 9999) is standalone: untouched, and cuts the chain.
    check("tree sample cut", {100, -80, 60}, {0, 1, 1}, d, {100, -80, 60}, {10, 9999, 12});

    // `needed` only selects which entries get filtered; all others still count
    // as neighbours (skipping for distribution reasons does not exclude them).
    checkEntries("needed mask", makeEntries({100, -80, -500, 60, -40}), {0, 1, 1, 1, 1}, {d, true},
                 {expect(d, {{0, 100}, {1, -80}, {2, -500}, {3, 60}, {4, -40}}),
                  expect(d, {{0, -80}, {1, -500}, {2, 60}, {3, -40}}),
                  -500,
                  expect(d, {{0, 60}, {1, -40}}),
                  -40},
                 {1, 1, 0, 1, 1});

    // Placeholder zero (decisive game, previous |score| > 100) is left out,
    // but its ply still counts: 300 is 2 plies ahead and gets d^2. (Its own
    // filtered value does not matter, the skip predicate drops it.)
    checkEntries("placeholder zero", makeEntries({500, 0, 300}), {0, 1, 1}, {d, true},
                 {expect(d, {{0, 500}, {2, 300}}), expect(d, {{0, 0}, {1, 300}}), 300});
    // A zero in a drawn game is a real score.
    checkEntries("drawn zero", makeEntries({500, 0, 300}, {}, {0, 0, 0}), {0, 1, 1}, {d, true},
                 {expect(d, {{0, 500}, {1, 0}, {2, 300}}), expect(d, {{0, 0}, {1, 300}}), 300});
    // A zero after a small score is a real score.
    checkEntries("real zero", makeEntries({50, 0, 300}), {0, 1, 1}, {d, true},
                 {expect(d, {{0, 50}, {1, 0}, {2, 300}}), expect(d, {{0, 0}, {1, 300}}), 300});

    // Position in check: left out with excludeTactical, counted without.
    {
        auto e = makeEntries({100, -80, 60});
        e[1].pos = chess::Position::fromFen(kCheckFen);
        checkEntries("in check excluded", e, {0, 1, 1}, {d, true}, {expect(d, {{0, 100}, {2, 60}}), -80, 60}, {1, 0, 1});
        checkEntries("in check counted", e, {0, 1, 1}, {d, false},
                     {expect(d, {{0, 100}, {1, -80}, {2, 60}}), -80, 60}, {1, 0, 1});
    }
    // Capturing move: likewise.
    {
        auto e = makeEntries({100, -80, 60});
        e[1].pos = chess::Position::fromFen(kCaptureFen);
        e[1].move = chess::Move{chess::e4, chess::d5};
        checkEntries("capture excluded", e, {0, 1, 1}, {d, true}, {expect(d, {{0, 100}, {2, 60}}), -80, 60}, {1, 0, 1});
    }

    // Hard cutoff after 10 plies: a big score 11 plies ahead does not reach
    // entry 0, but 10 plies ahead of entry 1 it does.
    {
        std::vector<int> scores;
        std::vector<char> cont;
        for (int i = 0; i < 13; ++i)
        {
            scores.push_back(i % 2 == 0 ? 100 : -100);
            cont.push_back(i > 0);
        }
        scores[11] = -3000;  // +3000 from the view of the even entries
        std::vector<std::pair<int, int>> t1;
        for (int k = 0; k <= 10; ++k) t1.push_back({k, scores[1 + k]});
        std::vector<char> needed(13, 0);
        needed[0] = needed[1] = 1;
        auto e = makeEntries(scores);
        std::vector<int> expected(scores);
        expected[0] = 100;
        expected[1] = expect(0.9, t1);
        checkEntries("cutoff at 10 plies", e, cont, {0.9, true}, expected, needed);
    }

    // Huge (non-mate) scores stay far below the mate range.
    check("clamp", {30000, -30000}, {0, 1}, d, {expect(d, {{0, 30000}, {1, -30000}}), -30000});

    // Single entry and empty input.
    check("single", {123}, {0}, 0.7, {123});
    check("empty", {}, {}, 0.7, {});

    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("all passed\n");
    return 0;
}
