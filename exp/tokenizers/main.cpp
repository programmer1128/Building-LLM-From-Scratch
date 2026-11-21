#include "BPE_Tokenizer.h"
#include<iostream>

int main()
{
       BPETokenizer tokenizer;
       std::string data="low low low lower lowest";

       //  std::vector<std::string> tokens=tokenizer.preTokenization(data);

       //  tokenizer.buildTokenIDSeq(tokens);
       //  tokenizer.display();
       std::cout<<"hello\n";
       tokenizer.init(data);
       std::cout<<"hello\n";
       tokenizer.display();
       std::cout<<"Enter the string\n";
       std::string str;
       std::getline(std::cin,str);
       std::vector<uint32_t>tokenIDs=tokenizer.encode(str);

       for(const uint32_t& id:tokenIDs)
       {
             std::cout<<id<<" ";
       }
       std::cout<<"\n";

       std::cout<<tokenizer.decode(tokenIDs);
       return 0;
}
