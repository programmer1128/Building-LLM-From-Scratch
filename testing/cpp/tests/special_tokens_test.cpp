#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// Marker interpretation is controlled by the explicit allowSpecial switch.
TEST(SpecialTokens, PreservesOrdinaryTextAndAllowsRegisteredControls)
{
    BPETokenizer tokenizer;
    tokenizer.init("", 256);
    tokenizer.addSpecialTokens({"<|endoftext|>"});
    const std::string marker = "<|endoftext|>";
    const auto control = tokenizer.encode(marker, true);
    ASSERT_EQ(control.size(), 1);
    EXPECT_NE(control, tokenizer.encode(marker, false));
    EXPECT_EQ(tokenizer.decode(control), marker);
    EXPECT_EQ(tokenizer.decode(control, true), "");
    EXPECT_EQ(tokenizer.decode(tokenizer.encode(marker, false)), marker);
}
