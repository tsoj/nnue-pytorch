#pragma once

// Temporal score filter.
//
// Replaces the score of every kept position by an exponentially decaying
// average, taken in win-probability space, of its own score and the scores of
// up to MAX_PLIES following positions of the same game:
//
//     p'_t = sum_k d^k q_{t+k} / sum_k d^k,   k = 0..MAX_PLIES
//     q_{t+k} = P(s_{t+k})       for even k,
//               1 - P(s_{t+k})   for odd k   (scores are side-to-move relative)
//     P(s) = sigmoid(s / SCALE),  s'_t = SCALE * logit(p'_t)
//
// Averaging probabilities keeps a few huge scores from dominating. The hard
// cutoff after MAX_PLIES hedges against long games with large scores.
//
// Following positions whose score is unreliable are left out of the average:
// placeholder zeros (same heuristic as the skip predicate) and, with
// Config::excludeTactical (set from the `filtered` skip option), captures and
// positions in check. They add neither a score nor a weight, but the decay
// keeps counting plies, so a position k plies ahead always gets d^k.
// Positions the skip predicate drops only to shape the data distribution
// (random, early-ply, piece-count and WLD skipping) still count.
//
// The chain is cut at game boundaries and at entries without a usable score
// (VALUE_NONE, mate / tablebase range scores) and at search-tree samples
// (ply 9999); nothing beyond such an entry is used. A position whose own score
// is not usable, or that has no usable neighbour, keeps its raw score.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "binpack.h"

namespace score_filter {

    constexpr int VALUE_NONE = 32002;
    // VALUE_TB_WIN_IN_MAX_PLY of Stockfish (VALUE_MATE 32000 - MAX_PLY 246).
    // Scores with a larger magnitude are mate or tablebase scores.
    constexpr int MAX_REGULAR_SCORE = 31754 - 1;
    // Ply that marks search-tree samples (see training_data_loader.cpp).
    constexpr int TREE_SAMPLE_PLY = 9999;
    // Following plies taken into account at most.
    constexpr int MAX_PLIES = 10;
    // Score -> win probability scale, matches the loss's out-scaling (~380).
    constexpr double SCALE = 380.0;
    // Keeps the logit finite: |s'| <= SCALE * 20.7 ~ 7900, below the mate range.
    constexpr double MIN_PROBABILITY = 1e-9;

    struct Config
    {
        // Decay in [0, 1); <= 0 disables the filter.
        double decay = 0.0;
        // Leave captures and positions in check out of the averages.
        bool excludeTactical = false;

        [[nodiscard]] bool enabled() const { return decay > 0.0; }
    };

    [[nodiscard]] inline bool isRegularScore(int score)
    {
        return score != VALUE_NONE && score >= -MAX_REGULAR_SCORE && score <= MAX_REGULAR_SCORE;
    }

    [[nodiscard]] inline double probability(int score)
    {
        return 1.0 / (1.0 + std::exp(-score / SCALE));
    }

    // Same heuristic as the skip predicate: a 0 in a decisive game right after
    // a clearly non-zero score was written as a placeholder, not searched.
    // flags[i] = 1 for such entries.
    inline void markPlaceholderZeros(
        const std::vector<binpack::TrainingDataEntry>& entries,
        const std::vector<char>& isCont,
        std::vector<char>& flags)
    {
        flags.assign(entries.size(), 0);
        int last = VALUE_NONE;
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            if (i == 0 || !isCont[i])
                last = VALUE_NONE;
            const auto& e = entries[i];
            if (e.score == VALUE_NONE)
                continue;
            if (last != VALUE_NONE && std::abs(last) > 100 && e.result != 0 && e.score == 0)
            {
                flags[i] = 1;
                continue;
            }
            last = e.score;
        }
    }

    // entries:   consecutive entries of one or more games, in file order.
    // isCont[i]: entries[i] directly follows entries[i-1] in the same game.
    //            isCont[0] is ignored.
    // out:       resized to entries.size(); filtered scores.
    // needed:    optional; only entries with needed[i] != 0 are filtered (the
    //            others keep their raw score). Saves work for skipped entries.
    // selfShare: optional, for statistics; resized to entries.size(). Weight
    //            share of the entry's own score in its average (1 if it was not
    //            averaged), or -1 if it was not considered (not needed, or no
    //            usable score of its own).
    inline void apply(
        const std::vector<binpack::TrainingDataEntry>& entries,
        const std::vector<char>& isCont,
        const Config& cfg,
        std::vector<std::int16_t>& out,
        const std::vector<char>* needed = nullptr,
        std::vector<float>* selfShare = nullptr)
    {
        const std::size_t n = entries.size();
        out.resize(n);
        if (selfShare)
            selfShare->assign(n, -1.0f);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = static_cast<std::int16_t>(entries[i].score);
        if (!cfg.enabled())
            return;

        std::vector<char> placeholder;
        markPlaceholderZeros(entries, isCont, placeholder);

        auto usable = [&](std::size_t i) {
            return isRegularScore(entries[i].score) && entries[i].ply != TREE_SAMPLE_PLY;
        };
        auto excluded = [&](std::size_t i) {
            return placeholder[i]
                || (cfg.excludeTactical && (entries[i].isCapturingMove() || entries[i].isInCheck()));
        };

        for (std::size_t i = 0; i < n; ++i)
        {
            if ((needed && !(*needed)[i]) || !usable(i))
                continue;
            if (selfShare)
                (*selfShare)[i] = 1.0f;

            double sum = probability(entries[i].score);
            double weight = 1.0;
            double dk = 1.0;
            bool any = false;

            for (std::size_t k = 1; k <= MAX_PLIES && i + k < n; ++k)
            {
                const std::size_t j = i + k;
                if (!isCont[j] || !usable(j))
                    break;
                dk *= cfg.decay;
                if (excluded(j))
                    continue;
                const double q = probability(entries[j].score);
                sum += dk * (k % 2 == 0 ? q : 1.0 - q);
                weight += dk;
                any = true;
            }

            if (!any)
                continue;
            if (selfShare)
                (*selfShare)[i] = static_cast<float>(1.0 / weight);

            const double p = std::clamp(sum / weight, MIN_PROBABILITY, 1.0 - MIN_PROBABILITY);
            out[i] = static_cast<std::int16_t>(std::lround(SCALE * std::log(p / (1.0 - p))));
        }
    }

}  // namespace score_filter
