#include "DataLoader.h"
#include <cstddef>
#include <cstring>
#include <optional>
#include <algorithm>

void DataLoader::initialise_Start_Indices(size_t len)
{
     startIndices.clear();
     if(len<=max_len)
     {
         return;
     }
     //the start index array will hold all the valid start positions from where
     //we can have the input token ID row vectors.
     //the max_len field defines the max amount of rows in a single row. So we 
     //need to take upto pos+max_len<len positions

     //so the size of the vector array will be len-max_len
     const size_t count = 1 + (len - max_len - 1) / stride;
     startIndices.reserve(count);

     for (size_t k = 0; k < count; ++k) 
     {
         startIndices.emplace_back(k * stride);
     }
     reset();
}

std::optional<DataLoader::Batch> DataLoader::nextBatch()
{
     if(cursor>=startIndices.size())
     {
         return std::nullopt;
     }
     size_t remaining = startIndices.size() - cursor;
     if (remaining == 0)
     {
         return std::nullopt;
     }

     if (drop_last && remaining < batchSize)
     {
         return std::nullopt;
     }

     size_t currentBatchSize = std::min(batchSize, remaining);

     if (currentBatchSize > std::numeric_limits<size_t>::max() / max_len) 
     {
         throw std::overflow_error("Batch token count overflow");
     }

     Batch newBatch;
     newBatch.batchSize=currentBatchSize;
     newBatch.sequenceLength = max_len;
     //now the batch will be dependent on batch size and the max_len and the curr cursor position
     //so as to know from where the input and output vectors will begin

     //we will need to add batch_size row vectors for both the input and output
     //and each vector will have a len of max_len. so in flattened format they have
     //batch_size*max_len elements so we reserve them from beginning to prevent slow down
     //due to resizing
     size_t elementSize=currentBatchSize*max_len;
     newBatch.input.resize(elementSize);
     newBatch.target.resize(elementSize);

     //making the batch
     for(size_t batchNum=0; batchNum<currentBatchSize;batchNum++)
     {
         size_t startPos= cursor+batchNum;
        
         size_t shuffled_OffSet= startIndices[startPos];

         //now from shuffled offset we need max_len tokens for the input vector
         //and from shuffled offset+1 we need max_len tokens for the output vector

         //copy from memory for the input vector. Now we need the offset of max_len*batchNum
         //because after a row is appended, the next row should be appended from the next position
         //in the flattened array
         std::memcpy(newBatch.input.data()+(max_len*batchNum),
             tokenIDs.data()+shuffled_OffSet,max_len*sizeof(uint32_t));

         //copying the output data in the same way as the input data, just the offset will be different
         std::memcpy(newBatch.target.data()+(max_len*batchNum),
             tokenIDs.data()+shuffled_OffSet+1,max_len*sizeof(uint32_t));
     }
     //move the cursor for the next batch
     cursor+=currentBatchSize;

     //return the current newBatch
     return newBatch;
} 


void DataLoader::reset()
{
     //this method is used for shuffling the starting indices to ensure that every epoch, we get
     //a new shuffled set of start indices for ensuring the entirety of data during the training

     cursor=0;
     std::shuffle(startIndices.begin(), startIndices.end(), rng_);
}