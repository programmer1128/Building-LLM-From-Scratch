#include "../support/test_support.h"

#include <gtest/gtest.h>

using namespace tokenizer_test;

// Truncated payloads, invalid magic and unsupported versions fail explicitly.
TEST(Serialization, RejectsCorruptArtifacts)
{
    TemporaryDirectory temporary;
    const auto path = temporary.path / "native.abpe";
    BPETokenizer original;
    original.init("abab abab", 260);
    original.save(path.string());
    const auto artifact = readFile(path);
    auto wrongVersion = artifact;
    wrongVersion[4] = static_cast<char>(255);
    const std::vector<std::string> invalid = {artifact.substr(0, artifact.size() - 1),
                                              "XXXX" + artifact.substr(4), wrongVersion};
    for (const auto &bytes : invalid)
    {
        std::ofstream output(path, std::ios::binary);
        output.write(bytes.data(), bytes.size());
        output.close();
        BPETokenizer restored;
        EXPECT_THROW(restored.load(path.string()), std::runtime_error);
    }
}
