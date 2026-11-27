#pragma once
#include <ostream>
#include <vector>
#include <cstdint>
#include <optional>
#include <iostream>
#include <random>

class DataLoader
{
     private:
     size_t max_len;
     size_t stride;
     size_t batchSize;
     size_t cursor;
     bool drop_last;
     const std::vector<uint32_t>& tokenIDs;
     std::vector<size_t> startIndices;

     std::mt19937 rng_{std::random_device{}()};

     void initialise_Start_Indices(size_t len);

     public:

     //the Batch in 2D conceptual view will be like this
     /*
       _  X _     _ Y _
      |1 2 3 |   |7 8 9|  
      |      |   |     |
      |4 5 6 |   |1 4 5|
      |      |   |     |
      |7 8 9 |   |2 3 6|
      |_    _|   |_   _|

      X is the input and Y is the output to be produced
      */

     class Batch 
     {
         public:
         //flattened vector of 2D matrix containing rows of input tokenIDs
         std::vector<uint32_t> input;
         //flattened vector of 2D matrix containing rows of output tokenIDs
         std::vector<uint32_t> output;   

         std::pair<size_t,size_t> dimensions;
     };

     static void vecPrint(const std::vector<uint32_t>&vec, std::string_view flag,std::pair<size_t,size_t>&dim, 
         std::ostream& out)
     {
         out<<flag<<" Dimension = "<<dim.first<<"x"<<dim.second<<"\n";
         out<<"[\n";
         size_t counter=0;

         for(const uint32_t inp:vec)
         {
             out<<inp<<" ";
             counter++;
             if(counter>=dim.second)
             {
                 counter=0;
                 out<<"\n";
             }
         }
         out<<"]";
         out<<"\n\n";
     }

     friend std::ostream& operator<<(std::ostream& out, const Batch &batch)
     {
         std::pair<size_t,size_t>dim=batch.dimensions;
         vecPrint(batch.input, "Input", dim, out);
         vecPrint(batch.output, "Output", dim, out);

         return out;
     }

     DataLoader(const std::vector<uint32_t>& tokenIDs, size_t max_len, size_t stride, size_t batchSize,
     bool drop_last=true):tokenIDs(tokenIDs)
     {
         this->max_len=max_len;
         this->stride=stride;
         this->batchSize=batchSize;
         if (max_len == 0 || stride == 0 || batchSize == 0)
         {
             throw std::invalid_argument("DataLoader dimensions must be positive");
         }  
         if (max_len > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) 
         {
             throw std::overflow_error("max_len byte size overflow");
         }
         this->drop_last=drop_last;
         this->cursor=0;
         initialise_Start_Indices(tokenIDs.size());
     }

     void reset();


     //provides the next batch for testing
     std::optional<Batch> nextBatch();
};