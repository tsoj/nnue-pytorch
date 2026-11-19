// Statistics for calibrating the temporal score filter (data_loader/cpp/lib/score_filter.h).
//
// Reads the first N chunks of a .binpack file and, for a list of decay values,
// reports how the filtered scores relate to the raw ones and how well they
// predict the game result.
//
// Build (links the loader to reuse its real skip predicate):
//   g++ -std=c++20 -O3 -march=native -pthread -I data_loader/cpp -I data_loader/cpp/lib \
//       scripts/score_filter_stats.cpp data_loader/cpp/training_data_loader.cpp -o score_filter_stats
// Run:
//   ./score_filter_stats FILE.binpack [--chunks N] [--stride K] [--skip] [--decays a,b,c]
//     --chunks N    read N chunks (default 200)
//     --stride K    read every K-th chunk, to sample the whole file (default 1)
//     --skip        only count positions the training skip predicate accepts, using the
//                   skipping settings of threats.yaml (as in training, only positions with
//                   unreliable scores are left out of the averages). adj_noise then counts pairs whose first
//                   position was accepted.
//     --decays      comma separated decay list
//
// Columns:
//   best_scale   scale k (internal score units) minimizing the loss below
//   result_loss  mean logistic loss of sigmoid(score/k) against the game result
//                (win=1, draw=0.5, loss=0), from the side to move's view
//   adj_noise    mean |s_t + s_{t+1}| over adjacent positions of one game, i.e. how
//                much consecutive evaluations disagree (scores alternate sign)
//   mean|d|      mean |filtered - raw|
//   rms          root mean square of the filtered scores (shrinkage check)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "score_filter.h"
#include "training_data_loader_internal.h"

using binpack::TrainingDataEntry;

struct Acc
{
    double adjSum = 0; std::uint64_t adjN = 0;
    double absDiff = 0, sq = 0; std::uint64_t n = 0;
    std::vector<double> lossPerScale;
};

static const std::vector<double> kScales = {50, 75, 100, 150, 200, 300, 400, 600, 800, 1200};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s FILE.binpack [--chunks N] [--stride K] [--skip] [--decays a,b,c]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    std::size_t maxChunks = 200, stride = 1;
    bool useSkip = false;
    std::vector<double> decays;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--chunks" && i + 1 < argc) maxChunks = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--stride" && i + 1 < argc) stride = std::max<std::size_t>(1, std::strtoull(argv[++i], nullptr, 10));
        else if (a == "--skip") useSkip = true;
        else if (a == "--decays" && i + 1 < argc)
        {
            std::string list = argv[++i];
            for (std::size_t pos = 0; pos <= list.size();)
            {
                const auto comma = list.find(',', pos);
                decays.push_back(std::atof(list.substr(pos, comma - pos).c_str()));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (decays.empty()) decays = {0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95};

    // Skipping settings of threats.yaml (common_run_options).
    std::function<bool(const TrainingDataEntry&)> skip;
    if (useSkip)
    {
        DataloaderSkipConfig c{};
        c.filtered = true;
        c.wld_filtered = true;
        c.random_fen_skipping = 3;
        c.early_fen_skipping = 18;
        c.soft_early_fen_skipping = 32;
        c.simple_eval_skipping = -1;
        c.param_index = 0;
        c.pc_y0 = -0.20; c.pc_y1 = 0.45; c.pc_y2 = 1.0; c.pc_y3 = 0.95; c.pc_y4 = 0.75;
        c.ply_x1 = 0.0;  c.ply_y1 = 0.025;
        c.ply_x2 = 22.0; c.ply_y2 = 0.05;
        c.ply_x3 = 25.5; c.ply_y3 = 0.20;
        c.ply_x4 = 29.5; c.ply_y4 = 0.80;
        c.score_filter_decay = 0.0;
        skip = make_skip_predicate(c);
    }

    binpack::CompressedTrainingDataFile file(path, std::ios_base::in);

    std::vector<Acc> acc(decays.size());
    for (auto& a : acc) a.lossPerScale.assign(kScales.size(), 0.0);

    std::vector<unsigned char> chunk;
    std::vector<TrainingDataEntry> entries;
    std::vector<char> isCont;
    std::vector<std::int16_t> filtered;

    std::uint64_t total = 0, contCount = 0, chains = 0, flagMismatch = 0, missedCont = 0;
    // Perspective check on raw scores of adjacent positions: if scores are side-to-move
    // relative they alternate sign, so |s_t + s_{t+1}| must be much smaller than |s_t - s_{t+1}|.
    double sumPlus = 0, sumMinus = 0; std::uint64_t pairN = 0;
    std::size_t chunksRead = 0;

    std::vector<char> accepted;
    std::uint64_t acceptedCount = 0;

    while (chunksRead < maxChunks && file.hasNextChunk())
    {
        file.readNextChunkInto(chunk);
        ++chunksRead;
        if (stride > 1)
        {
            std::size_t skipped = 0;
            (void) file.skipChunks(stride - 1, &skipped);
        }

        binpack::ChunkReader reader;
        entries.clear();
        isCont.clear();
        while (reader.hasNext(chunk))
        {
            isCont.push_back(reader.nextIsContinuation());
            entries.push_back(reader.next(chunk));
        }

        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            ++total;
            if (isCont[i]) ++contCount; else ++chains;
            if (i > 0)
            {
                if (isCont[i] && score_filter::isRegularScore(entries[i].score)
                    && score_filter::isRegularScore(entries[i - 1].score))
                {
                    sumPlus += std::abs(entries[i - 1].score + entries[i].score);
                    sumMinus += std::abs(entries[i - 1].score - entries[i].score);
                    ++pairN;
                }
                const bool real = binpack::isContinuation(entries[i - 1], entries[i]);
                if (isCont[i] && !real) ++flagMismatch;   // flagged but not a real successor
                if (!isCont[i] && real) ++missedCont;     // real successor stored as new chain
            }
        }

        // The skip predicate is stateful and sees entries in file order, exactly once.
        accepted.assign(entries.size(), 1);
        if (skip)
            for (std::size_t i = 0; i < entries.size(); ++i)
                accepted[i] = !skip(entries[i]);
        for (char a : accepted) acceptedCount += a;

        for (std::size_t di = 0; di < decays.size(); ++di)
        {
            score_filter::apply(entries, isCont, score_filter::Config{decays[di], true}, filtered, &accepted);
            Acc& a = acc[di];
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                if (!accepted[i]) continue;
                const int raw = entries[i].score;
                if (!score_filter::isRegularScore(raw)) continue;
                const int f = filtered[i];
                a.absDiff += std::abs(f - raw);
                a.sq += double(f) * f;
                ++a.n;

                if (i + 1 < entries.size() && isCont[i + 1]
                    && score_filter::isRegularScore(entries[i + 1].score))
                {
                    a.adjSum += std::abs(f + int(filtered[i + 1]));
                    ++a.adjN;
                }

                const double t = (entries[i].result + 1) * 0.5;
                for (std::size_t k = 0; k < kScales.size(); ++k)
                {
                    const double x = f / kScales[k];
                    // stable -[t ln s(x) + (1-t) ln s(-x)]
                    const double lp = -std::log1p(std::exp(-std::abs(x))) + std::min(x, 0.0);
                    const double ln = -std::log1p(std::exp(-std::abs(x))) + std::min(-x, 0.0);
                    a.lossPerScale[k] += -(t * lp + (1 - t) * ln);
                }
            }
        }
    }

    std::printf("file: %s\nchunks: %zu (stride %zu)  positions: %llu  chains: %llu  mean chain length: %.1f\n",
                path.c_str(), chunksRead, stride, (unsigned long long) total, (unsigned long long) chains,
                chains ? double(total) / chains : 0.0);
    std::printf("skip predicate: %s  accepted positions: %llu (%.1f%%)\n", useSkip ? "threats.yaml settings" : "off",
                (unsigned long long) acceptedCount, total ? 100.0 * acceptedCount / total : 0.0);
    if (pairN)
        std::printf("perspective check over %llu adjacent pairs: mean|s_t + s_t+1| = %.1f  mean|s_t - s_t+1| = %.1f  -> %s\n",
                    (unsigned long long) pairN, sumPlus / pairN, sumMinus / pairN,
                    sumPlus < sumMinus ? "side-to-move relative (as assumed)" : "NOT side-to-move relative!");
    std::printf("continuation flag check: flagged-but-not-real=%llu  real-but-not-flagged=%llu\n\n",
                (unsigned long long) flagMismatch, (unsigned long long) missedCont);

    std::printf("%8s %11s %12s %10s %9s %9s\n", "decay", "best_scale", "result_loss", "adj_noise", "mean|d|", "rms");
    for (std::size_t di = 0; di < decays.size(); ++di)
    {
        const Acc& a = acc[di];
        if (!a.n) continue;
        std::size_t best = 0;
        for (std::size_t k = 1; k < kScales.size(); ++k)
            if (a.lossPerScale[k] < a.lossPerScale[best]) best = k;
        std::printf("%8.3f %11.0f %12.5f %10.2f %9.2f %9.2f\n", decays[di], kScales[best],
                    a.lossPerScale[best] / a.n, a.adjN ? a.adjSum / a.adjN : 0.0,
                    a.absDiff / a.n, std::sqrt(a.sq / a.n));
    }
    return 0;
}
