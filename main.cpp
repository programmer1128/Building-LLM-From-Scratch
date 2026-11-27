#include "exp/tokenizers/BPE_Tokenizer.h"
#include "exp/DataLoaders/DataLoader.h"
#include <iostream>


int main()
{
     BPETokenizer tokenizer;
     std::string data="low low low lower lowest";

     tokenizer.init(data);
     tokenizer.display();
     std::cout<<"Enter the string\n";
     std::string str;
     std::getline(std::cin,str);
     std::vector<uint32_t>tokenIDs=tokenizer.encode(str);

     for(const uint32_t& id:tokenIDs)
     {
         std::cout<<id<<" ";
     }
     std::cout<<"\n\n";

     std::cout<<tokenizer.decode(tokenIDs)<<"\n";

     DataLoader dataLoader(tokenIDs,3,1,3);


     //simple training loop for 2 epochs
     for(size_t epoch=0; epoch<2;epoch++)
     {
         std::cout<<"epoch = "<<epoch<<"\n";
         while(auto batch = dataLoader.nextBatch())
         {
             std::cout<<*batch;
         }
         //reset for the next epoch
         dataLoader.reset();
     }

     return 0;
}