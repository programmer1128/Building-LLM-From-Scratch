#include "../../../exp/tokenizers/SplitStateMachine.h"
#include "../support/test_support.h"

#include <chrono>
#include <dlfcn.h>
#include <iostream>

using namespace tokenizer_test;
using Clock = std::chrono::steady_clock;

static size_t peakKiB()
{
    std::ifstream status("/proc/self/status");
    for (std::string line; std::getline(status, line);)
    {
        if (line.starts_with("VmHWM:"))
        {
            return std::stoull(line.substr(6));
        }
    }
    throw std::runtime_error("VmHWM unavailable");
}

static void writeIds(const Ids &ids)
{
    writeWord(std::cout, ids.size());
    for (const auto id : ids)
    {
        writeWord(std::cout, id);
    }
}

static void measure(const BPETokenizer &engine, char **argv)
{
    const auto input = readFile(argv[3]);
    const int passes = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
    const std::string op = argv[6];
    const bool decode = op == "decode";
    Ids prepared;
    if (decode)
    {
        prepared = engine.encode(input, false);
    }
    auto profileStart = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "token_profile_start"));
    auto profileStop =
        reinterpret_cast<void (*)(uint64_t *)>(dlsym(RTLD_DEFAULT, "token_profile_stop"));
    auto run = [&]
    {
        if (decode)
        {
            auto result = engine.decode(prepared);
            require(result.size() == input.size(), "incorrect decoded size");
            return prepared.size();
        }
        return engine.encode(input, false).size();
    };
    for (int r = -1; r < repeats; ++r)
    {
        if (r < 0 && input.size() >= 100'000'000)
        {
            continue;
        }
        uint64_t counters[8]{};
        if (profileStart)
        {
            profileStart();
        }
        const auto start = Clock::now();
        size_t count = 0;
        const int iterations = r < 0 && input.size() > 1'000'000 ? 1 : passes;
        for (int i = 0; i < iterations; ++i)
        {
            count += run();
        }
        const auto sec = std::chrono::duration<double>(Clock::now() - start).count();
        if (profileStop)
        {
            profileStop(counters);
        }
        if (r < 0)
        {
            continue;
        }
        std::cout << "{\"seconds\":" << sec << ",\"tokens\":" << count
                  << ",\"peak_kib\":" << peakKiB();
        if (profileStop)
        {
            std::cout << ",\"profile\":[";
            for (int i = 0; i < 8; ++i)
            {
                std::cout << (i ? "," : "") << counters[i];
            }
            std::cout << ']';
        }
        std::cout << "}\n" << std::flush;
    }
}

int main(int argc, char **argv)
{
    try
    {
        require(argc >= 3, "Expected mode and rank fixture");
        const std::string mode = argv[1];
        const ImportedRanks imported(argv[2]);
        const auto &engine = imported.engine;
        if (mode == "bench")
        {
            require(argc == 7, "Expected file, passes, repeats and operation");
            measure(engine, argv);
        }
        else if (mode == "diff" || mode == "specials" || mode == "splits")
        {
            while (std::cin.peek() != EOF)
            {
                const auto input = readBytes(std::cin);
                if (mode == "splits")
                {
                    const auto spans = SplitStateMachine::split(input);
                    writeWord(std::cout, spans.size());
                    for (const auto span : spans)
                    {
                        writeWord(std::cout, span.size());
                    }
                }
                else
                {
                    auto ids = engine.encode(input, mode == "specials");
                    require(engine.decode(ids) == input, "C++ byte round trip failed");
                    imported.toOfficial(ids);
                    writeIds(ids);
                }
            }
        }
        else
        {
            throw std::invalid_argument("Unknown adapter operation");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Tokenizer adapter failed: " << error.what() << '\n';
        return 1;
    }
}
