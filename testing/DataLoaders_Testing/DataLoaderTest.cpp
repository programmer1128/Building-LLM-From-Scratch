#include <gtest/gtest.h>
#include "../../DataLoaders/DataLoader.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    using RowPair = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

    // Helper to unpack a flattened Batch into (input_row, output_row) pairs
    // so tests can verify exact input-target pairing regardless of row shuffle order.
    std::vector<RowPair> extractRowPairs(const DataLoader::Batch& batch)
    {
        const size_t rows = batch.batchSize;
        const size_t cols = batch.sequenceLength;

        std::vector<RowPair> pairs;
        pairs.reserve(rows);

        for (size_t r = 0; r < rows; ++r)
        {
            const size_t start = r * cols;
            std::vector<uint32_t> x(batch.input.begin() + start, batch.input.begin() + start + cols);
            std::vector<uint32_t> y(batch.target.begin() + start, batch.target.begin() + start + cols);
            pairs.emplace_back(std::move(x), std::move(y));
        }

        return pairs;
    }
}

// ============================================================================
// TEST 1: Invalid Zero Dimensions
// What is verified:
//   Constructor rejects max_len == 0, stride == 0, or batchSize == 0 to prevent
//   infinite loops, zero-advance cursors, or meaningless 0-token sequences.
// How it is verified:
//   Instantiates DataLoader three times with each dimension set to 0 and
//   asserts that std::invalid_argument is thrown in every case.
// Data generated:
//   A small valid token vector {10, 20, 30, 40, 50} (length = 5).
// ============================================================================
TEST(DataLoaderEdgeCasesTest, InvalidZeroDimensions_ThrowsInvalidArgument)
{
    const std::vector<uint32_t> tokens = {10, 20, 30, 40, 50};

    EXPECT_THROW(DataLoader(tokens, 0, 1, 2, true), std::invalid_argument);
    EXPECT_THROW(DataLoader(tokens, 3, 0, 2, true), std::invalid_argument);
    EXPECT_THROW(DataLoader(tokens, 3, 1, 0, true), std::invalid_argument);
}

// ============================================================================
// TEST 2: Empty Dataset
// What is verified:
//   Passing an empty token vector does not trigger unsigned size_t underflow
//   during index reservation and safely yields zero batches.
// How it is verified:
//   Constructs DataLoader with an empty vector and both drop_last modes,
//   verifying that nextBatch() immediately returns std::nullopt.
// Data generated:
//   Empty std::vector<uint32_t> {} (length = 0), max_len = 4, stride = 1, batchSize = 2.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, EmptyDataset_ReturnsNulloptImmediately)
{
    const std::vector<uint32_t> emptyTokens = {};

    DataLoader loaderDrop(emptyTokens, 4, 1, 2, true);
    EXPECT_FALSE(loaderDrop.nextBatch().has_value());

    DataLoader loaderKeep(emptyTokens, 4, 1, 2, false);
    EXPECT_FALSE(loaderKeep.nextBatch().has_value());
}

// ============================================================================
// TEST 3: Dataset Length Equal to Context Length (L == max_len)
// What is verified:
//   When token count equals max_len, no shifted target sequence exists (the
//   final input token has no next token), so 0 valid samples are created.
// How it is verified:
//   Constructs DataLoader with L = 4 and max_len = 4, and asserts that
//   nextBatch() returns std::nullopt without reading out of bounds.
// Data generated:
//   4-token sequence {100, 200, 300, 400} (length = 4), max_len = 4, stride = 1, batchSize = 1.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, DatasetEqualToMaxLen_ReturnsNullopt)
{
    const std::vector<uint32_t> tokens = {100, 200, 300, 400};

    DataLoader loader(tokens, 4, 1, 1, false);
    EXPECT_FALSE(loader.nextBatch().has_value());
}

// ============================================================================
// TEST 4: Single Valid Sequence (L == max_len + 1)
// What is verified:
//   Minimum viable corpus produces 1 sequence; drop_last=true discards it when
//   batchSize > 1, while drop_last=false yields a 1x3 batch with exact shift.
// How it is verified:
//   Checks std::nullopt for drop_last=true (batchSize=2), then inspects dimensions
//   (1x3), buffer sizes (3), and exact input/output token values for drop_last=false.
// Data generated:
//   4-token sequence {10, 20, 30, 40} (length = 4), max_len = 3, stride = 1, batchSize = 2.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, SingleValidSequence_RespectsDropLastPolicy)
{
    const std::vector<uint32_t> tokens = {10, 20, 30, 40};

    DataLoader loaderDrop(tokens, 3, 1, 2, true);
    EXPECT_FALSE(loaderDrop.nextBatch().has_value());

    DataLoader loaderKeep(tokens, 3, 1, 2, false);
    const auto batchOpt = loaderKeep.nextBatch();
    ASSERT_TRUE(batchOpt.has_value());

    EXPECT_EQ(batchOpt->batchSize, 1);
    EXPECT_EQ(batchOpt->sequenceLength, 3);
    EXPECT_EQ(batchOpt->input.size(), 3);
    EXPECT_EQ(batchOpt->target.size(), 3);

    const std::vector<uint32_t> expectedInput = {10, 20, 30};
    const std::vector<uint32_t> expectedOutput = {20, 30, 40};
    EXPECT_EQ(batchOpt->input, expectedInput);
    EXPECT_EQ(batchOpt->target, expectedOutput);

    EXPECT_FALSE(loaderKeep.nextBatch().has_value());
}

// ============================================================================
// TEST 5: Exact Full Batches (No Off-By-One Final Batch Drop)
// What is verified:
//   When total valid windows is an exact multiple of batchSize (6 samples,
//   batchSize = 3), both batches are returned and none are dropped.
// How it is verified:
//   Drains all batches from the loader, verifies exact batch count (2), checks
//   3x3 dimensions and 9-element buffer sizes, and verifies all 6 expected
//   (input, target) pairs are present without duplicates or omissions.
// Data generated:
//   9 sequential tokens {1, 2, 3, 4, 5, 6, 7, 8, 9} (L = 9), max_len = 3, stride = 1.
//   Valid windows: 1 + (9 - 3 - 1) / 1 = 6 windows -> 2 full batches of 3.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, ExactFullBatches_YieldsAllBatchesWithoutDroppingLast)
{
    const std::vector<uint32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    DataLoader loader(tokens, 3, 1, 3, true);

    std::vector<RowPair> collectedPairs;
    size_t batchCount = 0;

    while (auto batch = loader.nextBatch())
    {
        ++batchCount;
        EXPECT_EQ(batch->batchSize, 3);
        EXPECT_EQ(batch->sequenceLength, 3);
        EXPECT_EQ(batch->input.size(), 9);
        EXPECT_EQ(batch->target.size(), 9);

        auto rows = extractRowPairs(*batch);
        collectedPairs.insert(collectedPairs.end(), rows.begin(), rows.end());
    }

    ASSERT_EQ(batchCount, 2);
    ASSERT_EQ(collectedPairs.size(), 6);

    std::vector<RowPair> expectedPairs = {
        {{1, 2, 3}, {2, 3, 4}},
        {{2, 3, 4}, {3, 4, 5}},
        {{3, 4, 5}, {4, 5, 6}},
        {{4, 5, 6}, {5, 6, 7}},
        {{5, 6, 7}, {6, 7, 8}},
        {{6, 7, 8}, {7, 8, 9}}
    };

    std::sort(collectedPairs.begin(), collectedPairs.end());
    std::sort(expectedPairs.begin(), expectedPairs.end());
    EXPECT_EQ(collectedPairs, expectedPairs);
}

// ============================================================================
// TEST 6: Partial Tail Batch With drop_last = true vs. drop_last = false
// What is verified:
//   When 7 valid windows are divided into batches of 3:
//   - drop_last = true yields 2 full (3x3) batches (6 rows total).
//   - drop_last = false yields 2 full (3x3) batches plus 1 partial (1x3) batch
//     with exact resized buffers (3 elements, not 9) and covers all 7 windows.
// How it is verified:
//   Runs both configurations on the same 10-token dataset, asserts batch counts,
//   per-batch row dimensions, buffer lengths, and complete window coverage.
// Data generated:
//   10 tokens {10, 20, 30, 40, 20, 50, 60, 20, 70, 80} (L = 10), max_len = 3, stride = 1.
//   Valid windows: 1 + (10 - 3 - 1) / 1 = 7 windows.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, PartialTailBatch_HandlesDropLastEnabledAndDisabled)
{
    const std::vector<uint32_t> tokens = {10, 20, 30, 40, 20, 50, 60, 20, 70, 80};

    // Case A: drop_last = true -> 2 batches of 3x3
    DataLoader loaderDrop(tokens, 3, 1, 3, true);
    size_t dropBatchCount = 0;
    while (auto batch = loaderDrop.nextBatch())
    {
        ++dropBatchCount;
        EXPECT_EQ(batch->batchSize, 3);
        EXPECT_EQ(batch->sequenceLength, 3);
        EXPECT_EQ(batch->input.size(), 9);
        EXPECT_EQ(batch->target.size(), 9);
    }
    EXPECT_EQ(dropBatchCount, 2);

    // Case B: drop_last = false -> 2 batches of 3x3 + 1 batch of 1x3
    DataLoader loaderKeep(tokens, 3, 1, 3, false);
    std::vector<size_t> observedRowCounts;
    std::vector<RowPair> allPairs;

    while (auto batch = loaderKeep.nextBatch())
    {
        observedRowCounts.push_back(batch->batchSize);
        EXPECT_EQ(batch->sequenceLength, 3);
        EXPECT_EQ(batch->input.size(), batch->batchSize * 3);
        EXPECT_EQ(batch->target.size(), batch->batchSize * 3);

        auto rows = extractRowPairs(*batch);
        allPairs.insert(allPairs.end(), rows.begin(), rows.end());
    }

    const std::vector<size_t> expectedRowCounts = {3, 3, 1};
    EXPECT_EQ(observedRowCounts, expectedRowCounts);
    ASSERT_EQ(allPairs.size(), 7);

    std::vector<RowPair> expectedAllPairs = {
        {{10, 20, 30}, {20, 30, 40}},
        {{20, 30, 40}, {30, 40, 20}},
        {{30, 40, 20}, {40, 20, 50}},
        {{40, 20, 50}, {20, 50, 60}},
        {{20, 50, 60}, {50, 60, 20}},
        {{50, 60, 20}, {60, 20, 70}},
        {{60, 20, 70}, {20, 70, 80}}
    };

    std::sort(allPairs.begin(), allPairs.end());
    std::sort(expectedAllPairs.begin(), expectedAllPairs.end());
    EXPECT_EQ(allPairs, expectedAllPairs);
}

// ============================================================================
// TEST 7: Non-Overlapping Stride (stride == max_len) and Target Alignment
// What is verified:
//   When stride > 1 (specifically stride = 4, max_len = 4 on 13 tokens), start
//   offsets advance by 4 without overlap, using token[12] as the final target.
// How it is verified:
//   Collects the single 3x4 batch and verifies that the 3 extracted (input, output)
//   row pairs match offsets 0, 4, and 8 exactly.
// Data generated:
//   13 sequential tokens {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} (L = 13),
//   max_len = 4, stride = 4, batchSize = 3, drop_last = true.
//   Valid windows: 1 + (13 - 4 - 1) / 4 = 3 windows.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, NonUnitStride_ComputesExactNonOverlappingWindows)
{
    const std::vector<uint32_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    DataLoader loader(tokens, 4, 4, 3, true);

    const auto batchOpt = loader.nextBatch();
    ASSERT_TRUE(batchOpt.has_value());
    EXPECT_EQ(batchOpt->batchSize, 3);
    EXPECT_EQ(batchOpt->sequenceLength, 4);

    std::vector<RowPair> actualPairs = extractRowPairs(*batchOpt);
    std::vector<RowPair> expectedPairs = {
        {{0, 1, 2, 3}, {1, 2, 3, 4}},
        {{4, 5, 6, 7}, {5, 6, 7, 8}},
        {{8, 9, 10, 11}, {9, 10, 11, 12}}
    };

    std::sort(actualPairs.begin(), actualPairs.end());
    std::sort(expectedPairs.begin(), expectedPairs.end());
    EXPECT_EQ(actualPairs, expectedPairs);

    EXPECT_FALSE(loader.nextBatch().has_value());
}

// ============================================================================
// TEST 8: Multi-Epoch Reset Reproducibility & Completeness
// What is verified:
//   Calling reset() at the end of an epoch resets cursor_ to 0 and allows a
//   complete second traversal over the exact same set of windows without losing
//   or corrupting any startIndices entries.
// How it is verified:
//   Iterates through 2 full epochs on a 6-window dataset, checking that both
//   epochs produce 2 full batches and contain the exact same 6 row pairs.
// Data generated:
//   9 tokens {11, 22, 33, 44, 55, 66, 77, 88, 99} (L = 9), max_len = 3, stride = 1, batchSize = 3.
// ============================================================================
TEST(DataLoaderEdgeCasesTest, ResetAcrossEpochs_PreservesCompleteSampleSet)
{
    const std::vector<uint32_t> tokens = {11, 22, 33, 44, 55, 66, 77, 88, 99};

    DataLoader loader(tokens, 3, 1, 3, true);

    std::vector<std::vector<RowPair>> epochPairs(2);

    for (size_t epoch = 0; epoch < 2; ++epoch)
    {
        while (auto batch = loader.nextBatch())
        {
            auto rows = extractRowPairs(*batch);
            epochPairs[epoch].insert(epochPairs[epoch].end(), rows.begin(), rows.end());
        }
        loader.reset();
    }

    ASSERT_EQ(epochPairs[0].size(), 6);
    ASSERT_EQ(epochPairs[1].size(), 6);

    std::sort(epochPairs[0].begin(), epochPairs[0].end());
    std::sort(epochPairs[1].begin(), epochPairs[1].end());
    EXPECT_EQ(epochPairs[0], epochPairs[1]);
}