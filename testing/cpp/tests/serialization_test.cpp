#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// Trained native vocabularies retain IDs and deterministic artifact bytes.
TEST(Serialization, PreservesTrainedVocabulary)
{
    TemporaryDirectory temporary;
    const auto first = temporary.path / "first.abpe";
    const auto second = temporary.path / "second.abpe";
    BPETokenizer original;
    original.init("low lower lowest low lower lowest", 300);
    original.addSpecialTokens({"<|endoftext|>"});
    original.save(first.string());
    BPETokenizer restored;
    restored.load(first.string());
    const std::string input = "lower <|endoftext|>";
    EXPECT_EQ(restored.encode(input), original.encode(input));
    EXPECT_EQ(restored.decode(restored.encode(input)), input);
    restored.save(second.string());
    EXPECT_EQ(readFile(first), readFile(second));
}
