#include "../support/test_support.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

using namespace tokenizer_test;

// Concurrent reads share the vocabulary and retain per-thread merge scratch.
TEST(Concurrency, SharedTokenizerHasStableOutputs)
{
    BPETokenizer tokenizer;
    tokenizer.init("hello hello world world বাংলা বাংলা", 300);
    const std::string input = "hello world বাংলা हिन्दी 🙂";
    const auto expected = tokenizer.encode(input, false);
    for (const unsigned workers : {2, 4, 8})
    {
        std::atomic<size_t> mismatches{0};
        std::vector<std::jthread> threads;
        for (unsigned worker = 0; worker < workers; ++worker)
        {
            threads.emplace_back(
                [&]()
                {
                    for (unsigned repeat = 0; repeat < 1000; ++repeat)
                    {
                        const auto ids = tokenizer.encode(input, false);
                        if (ids != expected || tokenizer.decode(ids) != input)
                        {
                            ++mismatches;
                        }
                    }
                });
        }
        threads.clear();
        EXPECT_EQ(mismatches.load(), 0) << "workers=" << workers;
    }
}
