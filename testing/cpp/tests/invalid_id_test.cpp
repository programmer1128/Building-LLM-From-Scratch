#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// Out-of-range IDs produce a documented exception rather than invalid access.
TEST(Decoder, RejectsInvalidId)
{
    BPETokenizer tokenizer;
    tokenizer.init("", 256);
    EXPECT_THROW(tokenizer.decode({UINT32_MAX}), std::out_of_range);
}
