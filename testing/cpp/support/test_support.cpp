#include "test_support.h"

#include <atomic>
#include <chrono>
#include <iterator>
#include <stdexcept>

namespace tokenizer_test
{
void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

uint32_t readWord(std::istream &input)
{
    std::array<unsigned char, 4> bytes{};
    require(static_cast<bool>(input.read(reinterpret_cast<char *>(bytes.data()), 4)),
            "Truncated uint32 record");
    return uint32_t{bytes[0]} | (uint32_t{bytes[1]} << 8) | (uint32_t{bytes[2]} << 16) |
           (uint32_t{bytes[3]} << 24);
}

void writeWord(std::ostream &output, uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
    {
        output.put(static_cast<char>((value >> shift) & 255));
    }
}

std::string readBytes(std::istream &input)
{
    const auto length = readWord(input);
    require(length <= 100'000'000, "Input record exceeds 100 MB");
    std::string result(length, '\0');
    require(static_cast<bool>(input.read(result.data(), length)), "Truncated byte record");
    return result;
}

std::string readFile(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Cannot open fixture");
    return {std::istreambuf_iterator<char>(input), {}};
}

ImportedRanks::ImportedRanks(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Cannot open rank fixture");
    const auto count = readWord(input);
    require(count == 100256, "Expected cl100k_base ranks");
    std::vector<std::string> pieces;
    pieces.reserve(count);
    for (uint32_t rank = 0; rank < count; ++rank)
    {
        auto piece = readBytes(input);
        if (rank < 256)
        {
            require(piece.size() == 1, "Expected base byte");
            officialByteIds[static_cast<unsigned char>(piece[0])] = rank;
        }
        pieces.push_back(std::move(piece));
    }

    const auto countSpecials = readWord(input);
    require(countSpecials == 5, "Expected five cl100k_base controls");
    std::vector<std::pair<std::string, uint32_t>> specials;
    for (uint32_t index = 0; index < countSpecials; ++index)
    {
        const auto id = readWord(input);
        specials.emplace_back(readBytes(input), id);
    }
    require(input.peek() == EOF, "Trailing rank data");
    engine.importRankedVocabulary(pieces, specials);
}

void ImportedRanks::toOfficial(Ids &ids) const
{
    for (auto &id : ids)
    {
        if (id < 256)
        {
            id = officialByteIds[id];
        }
    }
}

TemporaryDirectory::TemporaryDirectory()
{
    static std::atomic<unsigned> sequence{};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("tokenizer-test-" + std::to_string(stamp) + "-" + std::to_string(sequence++));
    require(std::filesystem::create_directory(path), "Cannot create test directory");
}

TemporaryDirectory::~TemporaryDirectory()
{
    std::error_code error;
    std::filesystem::remove_all(path, error);
}

std::vector<std::string> byteVocabulary()
{
    std::vector<std::string> pieces;
    for (unsigned value = 0; value < 256; ++value)
    {
        pieces.emplace_back(1, static_cast<char>(value));
    }
    return pieces;
}
} // namespace tokenizer_test
