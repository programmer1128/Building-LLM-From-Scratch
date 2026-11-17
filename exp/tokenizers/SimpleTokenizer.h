#pragma once
#include<unordered_map>
#include<cstdint>
#include<string>
#include <vector>
#include<unordered_set>
#include<set>

class Tokenizer
{
     private:
     std::unordered_map<std::string,uint32_t>vocabulary;    
     std::unordered_map<uint32_t,std::string>decoderMap;

     std::vector<std::string> tokenize(const std::string& data);
     void toPrint(std::set<std::string>& uniqueTokens);


      public:

      void init(const std::string& filePath);
      void buildVocabulary(const std::string& filePath);

      std::vector<std::string> decoder(const std::vector<uint32_t>&tokenIds,bool toPrint=false);

      std::vector<uint32_t> encoder(const std::vector<std::string>& data);
};