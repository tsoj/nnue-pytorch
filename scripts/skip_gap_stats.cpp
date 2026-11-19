// Distribution of the distance (in plies) between consecutive positions of the same
// game that survive the training skip predicate, with the skipping settings of
// nettest/threats.yaml (common_run_options).
//
// Build:
//   g++ -std=c++20 -O3 -march=native -pthread -I data_loader/cpp -I data_loader/cpp/lib \
//       scripts/skip_gap_stats.cpp data_loader/cpp/training_data_loader.cpp -o skip_gap_stats
// Run:
//   ./skip_gap_stats FILE.binpack [--chunks N] [--stride K]
//
// Game boundaries: a new game starts whenever the ply does not increase
// (entries of a game are stored in file order). Search-tree samples (ply 9999) are
// standalone and ignored.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "training_data_loader_internal.h"

using binpack::TrainingDataEntry;

static constexpr int TREE_SAMPLE_PLY = 9999;
static constexpr int MAXG = 512;

struct Hist
{
    std::vector<std::uint64_t> h = std::vector<std::uint64_t>(MAXG + 1, 0);
    std::uint64_t n = 0;
    double sum = 0;
    void add(int g)
    {
        h[std::min(g, MAXG)]++;
        ++n;
        sum += g;
    }
    int quantile(double q) const
    {
        std::uint64_t target = static_cast<std::uint64_t>(std::ceil(q * n)), c = 0;
        for (int g = 0; g <= MAXG; ++g)
            if ((c += h[g]) >= target) return g;
        return MAXG;
    }
    void print(const char* title) const
    {
        std::printf("\n== %s: n=%llu  mean=%.2f  p10=%d p25=%d median=%d p75=%d p90=%d p99=%d\n", title,
                    (unsigned long long) n, n ? sum / n : 0.0, quantile(0.10), quantile(0.25), quantile(0.5),
                    quantile(0.75), quantile(0.90), quantile(0.99));
        std::uint64_t cum = 0;
        int last = 0;
        for (int g = 0; g <= MAXG; ++g)
            if (h[g]) last = g;
        for (int g = 1; g <= std::min(last, 80); ++g)
        {
            cum += h[g];
            std::printf("%4d %10llu %6.2f%% %6.2f%%\n", g, (unsigned long long) h[g], 100.0 * h[g] / n,
                        100.0 * cum / n);
        }
        if (last > 80)
        {
            std::uint64_t rest = n - cum;
            std::printf(" >80 %10llu %6.2f%%\n", (unsigned long long) rest, 100.0 * rest / n);
        }
    }
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s FILE.binpack [--chunks N] [--stride K]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    std::size_t maxChunks = 2000, stride = 1;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--chunks" && i + 1 < argc) maxChunks = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--stride" && i + 1 < argc) stride = std::max<std::size_t>(1, std::strtoull(argv[++i], nullptr, 10));
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }

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
    c.tree_sample_accept = 0.12;
    c.score_filter_decay = 0.0;
    auto skip = make_skip_predicate(c);

    binpack::CompressedTrainingDataFile file(path, std::ios_base::in);
    std::vector<unsigned char> chunk;

    Hist rawGap, accGap, accPerGame;
    std::uint64_t total = 0, accepted = 0, games = 0, treeSamples = 0;
    std::size_t chunksRead = 0;

    int prevPly = -1;        // ply of the previous entry in file order
    int lastAccPly = -1;     // ply of the last accepted entry of the current game
    int accInGame = 0;

    auto endGame = [&] {
        if (prevPly >= 0) { accPerGame.add(accInGame); ++games; }
        lastAccPly = -1;
        accInGame = 0;
    };

    while (chunksRead < maxChunks && file.hasNextChunk())
    {
        file.readNextChunkInto(chunk);
        ++chunksRead;
        if (stride > 1)
        {
            std::size_t skipped = 0;
            (void) file.skipChunks(stride - 1, &skipped);
            endGame();       // games never span chunks, but do not join across skipped data
            prevPly = -1;
        }

        binpack::ChunkReader reader;
        while (reader.hasNext(chunk))
        {
            const TrainingDataEntry e = reader.next(chunk);
            ++total;
            // The predicate is stateful; it must see every entry in file order.
            const bool acc = !skip(e);

            if (e.ply == TREE_SAMPLE_PLY) { ++treeSamples; continue; }

            if (prevPly < 0 || e.ply <= prevPly)
                endGame();
            else
                rawGap.add(e.ply - prevPly);
            prevPly = e.ply;

            if (acc)
            {
                ++accepted;
                if (lastAccPly >= 0) accGap.add(e.ply - lastAccPly);
                lastAccPly = e.ply;
                ++accInGame;
            }
        }
        if (stride == 1) { /* chains never span chunks */ endGame(); prevPly = -1; }
    }
    endGame();

    std::printf("chunks=%zu entries=%llu tree_samples=%llu games=%llu accepted=%llu (%.2f%%)\n", chunksRead,
                (unsigned long long) total, (unsigned long long) treeSamples, (unsigned long long) games,
                (unsigned long long) accepted, 100.0 * accepted / total);
    rawGap.print("ply gap between consecutive stored entries of a game (before skipping)");
    accGap.print("ply gap between consecutive ACCEPTED positions of a game");
    accPerGame.h[0] = accPerGame.h[0];  // games with zero accepted positions are included
    accPerGame.print("accepted positions per game (row 0 omitted from table)");
    std::printf("games with 0 accepted: %llu (%.2f%%)\n", (unsigned long long) accPerGame.h[0],
                100.0 * accPerGame.h[0] / accPerGame.n);
    return 0;
}
