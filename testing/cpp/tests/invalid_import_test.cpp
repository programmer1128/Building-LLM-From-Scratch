#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// Failed imports must preserve the previously usable tokenizer.
TEST(RankedVocabulary, RejectsInvalidDataWithoutChangingState)
{
    BPETokenizer tokenizer;
    tokenizer.init("", 256);
    auto pieces = byteVocabulary();
    pieces[1] = pieces[0];
    EXPECT_THROW(tokenizer.importRankedVocabulary(pieces), std::invalid_argument);
    EXPECT_EQ(tokenizer.decode(tokenizer.encode("valid", false)), "valid");
    pieces = byteVocabulary();
    pieces.push_back("missing");
    EXPECT_THROW(tokenizer.importRankedVocabulary(pieces), std::invalid_argument);
}
