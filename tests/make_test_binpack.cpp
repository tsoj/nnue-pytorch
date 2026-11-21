// Generates small binpack files containing random games for data loader
// tests. Usage: make_test_binpack <output.binpack> <num_positions> [seed] [score]
// With [score], every entry gets that score instead of a random one.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "binpack.h"

using namespace binpack;

int main(int argc, char** argv) {
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <output.binpack> <num_positions> [seed] [score]\n", argv[0]);
        return 1;
    }

    const std::string output = argv[1];
    const long        target = std::atol(argv[2]);
    const unsigned    seed   = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 12345u;
    const bool        fixedScore = argc > 4;
    const int         score      = fixedScore ? std::atoi(argv[4]) : 0;
    std::mt19937      rng(seed);

    CompressedTrainingDataEntryWriter writer(output);

    long written = 0;
    while (written < target)
    {
        auto pos =
          chess::Position::fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        std::int16_t result = (rng() % 2) ? 1 : -1;

        for (int ply = 0; ply < 160 && written < target; ++ply)
        {
            std::vector<chess::Move> moves;
            chess::movegen::forEachLegalMove(pos, [&](chess::Move m) { moves.push_back(m); });
            if (moves.empty())
                break;

            TrainingDataEntry e;
            e.pos    = pos;
            e.move   = moves[rng() % moves.size()];
            const int randomScore = static_cast<int>(rng() % 2001) - 1000;
            e.score  = static_cast<std::int16_t>(fixedScore ? score : randomScore);
            e.ply    = static_cast<std::uint16_t>(ply);
            e.result = result;
            writer.addTrainingDataEntry(e);
            ++written;

            pos    = pos.afterMove(e.move);
            result = -result;
        }
    }

    return 0;
}
