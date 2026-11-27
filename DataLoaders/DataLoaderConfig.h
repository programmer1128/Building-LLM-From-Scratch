#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

struct DataLoaderConfig
{
     size_t sequenceLength = 0;
     size_t stride = 0;
     size_t batchSize = 0;
     bool shuffle = true;
     bool dropLast = false;
     uint64_t seed = 42;

     // Verifies all numeric invariants and overflow bounds.
     // Throws std::invalid_argument for invalid values and std::overflow_error on overflow.
     void validate() const;

     // Loads and validates a DataLoaderConfig from a .toml file on disk.
     // tablePath supports nested TOML paths such as "dataloader", "dataloader.train", or "".
     // Throws std::runtime_error on I/O or TOML syntax errors, std::invalid_argument on schema errors.
     static DataLoaderConfig fromTomlFile(const std::filesystem::path& filePath,
                                         std::string_view tablePath = "dataloader");

     // Loads and validates a DataLoaderConfig directly from an in-memory TOML string.
     static DataLoaderConfig fromTomlString(std::string_view tomlContent,
                                           std::string_view tablePath = "dataloader");

     bool operator==(const DataLoaderConfig& other) const = default;
};