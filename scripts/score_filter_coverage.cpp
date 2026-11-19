// How much the temporal score filter (data_loader/cpp/lib/score_filter.h)
// actually averages, on positions kept by the training skip predicate with the
// skipping settings of nettest/threats.yaml.
//
// For each decay: the share of kept positions that get averaged with at least
// one following position, and the weight share of a position's own score in
// its average (1 = not averaged at all).
//
// Build:
//   g++ -std=c++20 -O3 -march=native -pthread -I data_loader/cpp -I data_loader/cpp/lib \
//       scripts/score_filter_coverage.cpp data_loader/cpp/training_data_loader.cpp -o score_filter_coverage
// Run:
//   ./score_filter_coverage FILE.binpack [--chunks N] [--stride K] [--decays a,b,c]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "score_filter.h"
#include "training_data_loader_internal.h"

using binpack::TrainingDataEntry;

struct Acc
{
    std::uint64_t considered = 0;   // kept, with a usable own score
    std::uint64_t averaged = 0;     // ... and at least one contributing neighbour
    double shareSumAll = 0;         // own share over all considered (1 if not averaged)
    double shareSumAveraged = 0;    // own share over averaged ones only
    std::vector<std::uint64_t> bins = std::vector<std::uint64_t>(10, 0);  // share histogram
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s FILE.binpack [--chunks N] [--stride K] [--decays a,b,c]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    std::size_t maxChunks = 2000, stride = 1;
    std::vector<double> decays;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--chunks" && i + 1 < argc) maxChunks = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--stride" && i + 1 < argc) stride = std::max<std::size_t>(1, std::strtoull(argv[++i], nullptr, 10));
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
    if (decays.empty()) decays = {0.3, 0.5, 0.7, 0.8, 0.9, 0.95};

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
    std::vector<TrainingDataEntry> entries;
    std::vector<char> isCont, keep;
    std::vector<std::int16_t> out;
    std::vector<float> share;
    std::vector<Acc> acc(decays.size());
    std::uint64_t total = 0, kept = 0;
    std::size_t chunksRead = 0;

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
        // The skip predicate is stateful: every entry once, in file order.
        keep.resize(entries.size());
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            keep[i] = !skip(entries[i]);
            kept += keep[i];
        }
        total += entries.size();

        for (std::size_t di = 0; di < decays.size(); ++di)
        {
            score_filter::apply(entries, isCont, score_filter::Config{decays[di], c.filtered}, out, &keep, &share);
            Acc& a = acc[di];
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                const float s = share[i];
                if (s < 0) continue;
                ++a.considered;
                a.shareSumAll += s;
                if (s < 1.0f)
                {
                    ++a.averaged;
                    a.shareSumAveraged += s;
                }
                a.bins[std::min<std::size_t>(9, static_cast<std::size_t>(s * 10))]++;
            }
        }
    }

    std::printf("chunks=%zu (stride %zu) positions=%llu kept=%llu (%.2f%%)\n\n", chunksRead, stride,
                (unsigned long long) total, (unsigned long long) kept, 100.0 * kept / total);
    std::printf("%6s %10s %10s %13s %13s %11s\n", "decay", "considered", "averaged", "mean share", "mean share",
                "min share");
    std::printf("%6s %10s %10s %13s %13s %11s\n", "", "of kept", "", "(all)", "(averaged)", "possible");
    for (std::size_t di = 0; di < decays.size(); ++di)
    {
        const Acc& a = acc[di];
        double wmax = 0, dk = 1;
        for (int k = 0; k <= score_filter::MAX_PLIES; ++k) { wmax += dk; dk *= decays[di]; }
        std::printf("%6.2f %9.2f%% %9.2f%% %13.3f %13.3f %11.3f\n", decays[di], 100.0 * a.considered / kept,
                    100.0 * a.averaged / a.considered, a.shareSumAll / a.considered,
                    a.averaged ? a.shareSumAveraged / a.averaged : 0.0, 1.0 / wmax);
    }
    std::printf("\nshare histogram (fraction of considered positions; own share in [x, x+0.1), last bin incl. 1.0)\n%6s",
                "decay");
    for (int b = 0; b < 10; ++b) std::printf(" %6.1f", b / 10.0);
    std::printf("\n");
    for (std::size_t di = 0; di < decays.size(); ++di)
    {
        std::printf("%6.2f", decays[di]);
        for (int b = 0; b < 10; ++b) std::printf(" %5.1f%%", 100.0 * acc[di].bins[b] / acc[di].considered);
        std::printf("\n");
    }
    return 0;
}
