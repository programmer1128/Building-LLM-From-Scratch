#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// A permuted base alphabet still uses native byte IDs, with later rank IDs.
TEST(RankedVocabulary, ImportsPermutedBytesAndMerge)
{
    auto pieces = byteVocabulary();
    std::swap(pieces[0], pieces[255]);
    pieces.push_back("ab");
    BPETokenizer tokenizer;
    tokenizer.importRankedVocabulary(pieces);
    EXPECT_EQ(tokenizer.encode("ab", false), (Ids{256}));
    EXPECT_EQ(tokenizer.decode({256}), "ab");
    EXPECT_THROW(tokenizer.save("unused.abpe"), std::logic_error);
}
