#include "../support/test_support.h"

#include <gtest/gtest.h>

#include "../../../exp/tokenizers/SplitStateMachine.h"

using namespace tokenizer_test;

// Borrowed spans cover the original input in order without dropping whitespace.
TEST(SpanPartition, PreservesUnicodeWhitespaceAndNewlines)
{
    const std::string input = "  বাংলা हिन्दी 中文 🙂\r\n\t x";
    const auto spans = SplitStateMachine::split(input);
    std::string reconstructed;
    size_t offset = 0;
    for (const auto span : spans)
    {
        EXPECT_EQ(span.data(), input.data() + offset);
        reconstructed.append(span);
        offset += span.size();
    }
    EXPECT_EQ(reconstructed, input);
}
