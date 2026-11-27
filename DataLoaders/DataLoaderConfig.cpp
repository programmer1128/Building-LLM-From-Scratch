#include "DataLoaderConfig.h"

#include <toml++/toml.hpp>
#include <array>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
    constexpr std::array<std::string_view, 6> kAllowedKeys = {
        "sequence_length",
        "stride",
        "batch_size",
        "shuffle",
        "drop_last",
        "seed"
    };

    void rejectUnknownKeys(const toml::table& table, std::string_view tablePath)
    {
        for (const auto& [key, value] : table)
        {
            const std::string_view keyName = key.str();
            const bool isKnown = std::find(kAllowedKeys.begin(), kAllowedKeys.end(), keyName) != kAllowedKeys.end();

            if (!isKnown)
            {
                throw std::invalid_argument(
                    "Unknown key '" + std::string(keyName) +
                    "' in TOML table '" + std::string(tablePath) + "'"
                );
            }
        }
    }

    size_t parseRequiredPositiveSize(const toml::table& table,
                                     std::string_view key,
                                     std::string_view tablePath)
    {
        const toml::node* node = table.get(key);
        if (node == nullptr)
        {
            throw std::invalid_argument(
                "Missing required key '" + std::string(key) +
                "' in TOML table '" + std::string(tablePath) + "'"
            );
        }

        if (!node->is_integer())
        {
            throw std::invalid_argument(
                "Key '" + std::string(key) + "' must be an integer"
            );
        }

        const int64_t rawValue = node->as_integer()->get();
        if (rawValue <= 0)
        {
            throw std::invalid_argument(
                "Key '" + std::string(key) + "' must be a positive integer"
            );
        }

        const uint64_t unsignedValue = static_cast<uint64_t>(rawValue);
        if constexpr (sizeof(size_t) < sizeof(uint64_t))
        {
            if (unsignedValue > std::numeric_limits<size_t>::max())
            {
                throw std::overflow_error(
                    "Key '" + std::string(key) + "' exceeds size_t maximum"
                );
            }
        }

        return static_cast<size_t>(unsignedValue);
    }

    bool parseOptionalBool(const toml::table& table,
                           std::string_view key,
                           bool defaultValue)
    {
        const toml::node* node = table.get(key);
        if (node == nullptr)
        {
            return defaultValue;
        }

        if (!node->is_boolean())
        {
            throw std::invalid_argument(
                "Key '" + std::string(key) + "' must be a boolean (true or false)"
            );
        }

        return node->as_boolean()->get();
    }

    uint64_t parseOptionalNonNegativeUint64(const toml::table& table,
                                            std::string_view key,
                                            uint64_t defaultValue)
    {
        const toml::node* node = table.get(key);
        if (node == nullptr)
        {
            return defaultValue;
        }

        if (!node->is_integer())
        {
            throw std::invalid_argument(
                "Key '" + std::string(key) + "' must be an integer"
            );
        }

        const int64_t rawValue = node->as_integer()->get();
        if (rawValue < 0)
        {
            throw std::invalid_argument(
                "Key '" + std::string(key) + "' must be a non-negative integer"
            );
        }

        return static_cast<uint64_t>(rawValue);
    }

    DataLoaderConfig parseConfigFromRoot(const toml::table& root, std::string_view tablePath)
    {
        const toml::table* targetTable = nullptr;

        if (tablePath.empty())
        {
            targetTable = &root;
        }
        else
        {
            const toml::node_view<const toml::node> node = root.at_path(tablePath);
            if (!node)
            {
                throw std::invalid_argument(
                    "Missing TOML table '[" + std::string(tablePath) + "]'"
                );
            }
            if (!node.is_table())
            {
                throw std::invalid_argument(
                    "TOML path '" + std::string(tablePath) + "' is not a table"
                );
            }
            targetTable = node.as_table();
        }

        rejectUnknownKeys(*targetTable, tablePath);

        DataLoaderConfig config;
        config.sequenceLength = parseRequiredPositiveSize(*targetTable, "sequence_length", tablePath);
        config.stride = parseRequiredPositiveSize(*targetTable, "stride", tablePath);
        config.batchSize = parseRequiredPositiveSize(*targetTable, "batch_size", tablePath);
        config.shuffle = parseOptionalBool(*targetTable, "shuffle", config.shuffle);
        config.dropLast = parseOptionalBool(*targetTable, "drop_last", config.dropLast);
        config.seed = parseOptionalNonNegativeUint64(*targetTable, "seed", config.seed);

        config.validate();
        return config;
    }
}

void DataLoaderConfig::validate() const
{
    if (sequenceLength == 0 || stride == 0 || batchSize == 0)
    {
        throw std::invalid_argument(
            "DataLoaderConfig dimensions (sequenceLength, stride, batchSize) must be positive"
        );
    }

    if (sequenceLength > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
    {
        throw std::overflow_error("DataLoaderConfig sequenceLength byte size overflow");
    }

    if (batchSize > std::numeric_limits<size_t>::max() / sequenceLength)
    {
        throw std::overflow_error("DataLoaderConfig batch token count overflow");
    }
}

DataLoaderConfig DataLoaderConfig::fromTomlFile(const std::filesystem::path& filePath,
                                                std::string_view tablePath)
{
    if (!std::filesystem::exists(filePath))
    {
        throw std::runtime_error(
            "DataLoaderConfig file not found: " + filePath.string()
        );
    }

    try
    {
        const toml::table root = toml::parse_file(filePath.string());
        return parseConfigFromRoot(root, tablePath);
    }
    catch (const toml::parse_error& err)
    {
        throw std::runtime_error(
            "Failed to parse TOML file '" + filePath.string() + "': " + std::string(err.description())
        );
    }
}

DataLoaderConfig DataLoaderConfig::fromTomlString(std::string_view tomlContent,
                                                  std::string_view tablePath)
{
    try
    {
        const toml::table root = toml::parse(tomlContent);
        return parseConfigFromRoot(root, tablePath);
    }
    catch (const toml::parse_error& err)
    {
        throw std::runtime_error(
            "Failed to parse TOML content: " + std::string(err.description())
        );
    }
}