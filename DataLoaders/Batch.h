#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>
#include <vector>

// Represents a contiguous row-major micro-batch of shape (batchSize, sequenceLength).
// Flat 1D vectors are used for single heap allocation, CPU cache locality, and direct
// host-to-device (CUDA) memory transfers.
struct Batch
{
     std::vector<uint32_t> input;
     std::vector<uint32_t> target;
     size_t batchSize = 0;
     size_t sequenceLength = 0;

     size_t numTokens() const noexcept
     {
         return input.size();
     }

     bool empty() const noexcept
     {
         return input.empty();
     }
};

inline void printBatchMatrix(const std::vector<uint32_t>& vec,
                             std::string_view label,
                             size_t rows,
                             size_t cols,
                             std::ostream& out)
{
     out << label << " Dimension = " << rows << "x" << cols << "\n";
     out << "[\n";
     size_t counter = 0;

     for (const uint32_t tokenId : vec)
     {
         out << tokenId << " ";
         counter++;
         if (counter >= cols)
         {
             counter = 0;
             out << "\n";
        }
     }
     out << "]\n\n";
}

inline std::ostream& operator<<(std::ostream& out, const Batch& batch)
{ 
     printBatchMatrix(batch.input, "Input", batch.batchSize, batch.sequenceLength, out);
     printBatchMatrix(batch.target, "Target", batch.batchSize, batch.sequenceLength, out);
     return out;
}