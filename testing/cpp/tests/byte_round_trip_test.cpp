#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// Every raw byte, including NUL and invalid standalone UTF-8, must survive.
TEST(ByteRoundTrip, PreservesAllBytes)
{
    BPETokenizer tokenizer;
    tokenizer.init("", 256);
    for (unsigned value = 0; value < 256; ++value)
    {
        const std::string input(1, static_cast<char>(value));
        EXPECT_EQ(tokenizer.decode(tokenizer.encode(input, false)), input) << value;
    }
}
