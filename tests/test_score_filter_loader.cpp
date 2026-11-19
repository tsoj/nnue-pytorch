// End-to-end check of the score filter inside the parallel data loader:
// the multiset of scores coming out of the stream with decay d must equal the
// multiset obtained by running score_filter::apply over every chunk offline,
// with and without a skip predicate.
//
// Build & run:
//   g++ -std=c++20 -O2 -pthread -I data_loader/cpp/lib tests/test_score_filter_loader.cpp -o /tmp/t && /tmp/t .pgo/small.binpack

#include <algorithm>
#include <deque>
#include <functional>
#include <cstdio>
#include <string>
#include <vector>

#include "nnue_training_data_stream.h"
#include "score_filter.h"

// Deterministic, stateless skip predicate (the loader runs it on several threads).
static bool skipPred(const binpack::TrainingDataEntry& e)
{
    return (e.ply % 3) == 0 || (e.ply % 7) == 0;
}

static std::vector<int> offline(const std::string& path, score_filter::Config cfg, bool useSkip)
{
    binpack::CompressedTrainingDataFile file(path, std::ios_base::in);
    std::vector<unsigned char> chunk;
    std::vector<int> all;
    while (file.hasNextChunk())
    {
        file.readNextChunkInto(chunk);
        binpack::ChunkReader reader;
        std::vector<binpack::TrainingDataEntry> entries;
        std::vector<char> cont;
        while (reader.hasNext(chunk))
        {
            cont.push_back(reader.nextIsContinuation());
            entries.push_back(reader.next(chunk));
        }
        std::vector<char> keep(entries.size(), 1);
        if (useSkip)
            for (std::size_t i = 0; i < entries.size(); ++i)
                keep[i] = !skipPred(entries[i]);
        std::vector<std::int16_t> out;
        score_filter::apply(entries, cont, cfg, out, &keep);
        for (std::size_t i = 0; i < out.size(); ++i)
            if (keep[i]) all.push_back(out[i]);
    }
    std::sort(all.begin(), all.end());
    return all;
}

static std::vector<int> viaLoader(const std::string& path, score_filter::Config cfg, bool useSkip)
{
    std::function<bool(const binpack::TrainingDataEntry&)> pred;
    if (useSkip) pred = skipPred;
    auto stream = training_data::open_sfen_input_file_parallel(
        4, {path}, /*cyclic=*/false, pred, nullptr, 0, 1, {}, 2, cfg);
    std::vector<int> all;
    for (;;)
    {
        std::vector<binpack::TrainingDataEntry> v;
        stream->fill_threadsafe(v, 4096);
        for (auto& e : v) all.push_back(e.score);
        if (v.size() < 4096) break;
    }
    std::sort(all.begin(), all.end());
    return all;
}

int main(int argc, char** argv)
{
    const std::string path = argc > 1 ? argv[1] : ".pgo/small.binpack";
    int failures = 0;
    for (bool useSkip : {false, true})
        for (double d : {0.0, 0.5, 0.9})
        {
            const score_filter::Config cfg{d, true};
            const auto a = offline(path, cfg, useSkip);
            const auto b = viaLoader(path, cfg, useSkip);
            const bool ok = a == b;
            std::printf("%s skip=%d decay=%.2f offline=%zu loader=%zu\n", ok ? "ok  " : "FAIL", useSkip, d,
                        a.size(), b.size());
            failures += !ok;
        }
    // Excluding captures and checks must change the averages.
    {
        binpack::CompressedTrainingDataFile file(path, std::ios_base::in);
        std::vector<unsigned char> chunk;
        std::size_t differ = 0;
        while (file.hasNextChunk())
        {
            file.readNextChunkInto(chunk);
            binpack::ChunkReader reader;
            std::vector<binpack::TrainingDataEntry> entries;
            std::vector<char> cont;
            while (reader.hasNext(chunk))
            {
                cont.push_back(reader.nextIsContinuation());
                entries.push_back(reader.next(chunk));
            }
            std::vector<char> keep(entries.size());
            for (std::size_t i = 0; i < entries.size(); ++i) keep[i] = !skipPred(entries[i]);
            std::vector<std::int16_t> withMask, without;
            score_filter::apply(entries, cont, {0.9, true}, withMask, &keep);
            score_filter::apply(entries, cont, {0.9, false}, without, &keep);
            for (std::size_t i = 0; i < entries.size(); ++i) differ += keep[i] && withMask[i] != without[i];
        }
        const bool ok = differ > 0;
        std::printf("%s excluding captures/checks changes %zu kept scores\n", ok ? "ok  " : "FAIL", differ);
        failures += !ok;
    }
    return failures ? 1 : 0;
}
