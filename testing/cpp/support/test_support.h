#pragma once

#include "../../../exp/tokenizers/BPE_Tokenizer.h"

#include <array>
#include <filesystem>
#include <fstream>

namespace tokenizer_test
{
using Ids = std::vector<uint32_t>;

uint32_t readWord(std::istream &input);
void writeWord(std::ostream &output, uint32_t value);
std::string readBytes(std::istream &input);
std::string readFile(const std::filesystem::path &path);
void require(bool condition, const char *message);

// The adapter imports through public APIs and translates native base-byte IDs.
class ImportedRanks
{
  public:
    explicit ImportedRanks(const std::filesystem::path &path);
    void toOfficial(Ids &ids) const;
    BPETokenizer engine;

  private:
    std::array<uint32_t, 256> officialByteIds{};
};

// Each GoogleTest case owns an isolated artifact directory.
class TemporaryDirectory
{
  public:
    TemporaryDirectory();
    ~TemporaryDirectory();
    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
    std::filesystem::path path;
};

std::vector<std::string> byteVocabulary();
} // namespace tokenizer_test
