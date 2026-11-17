#pragma once
#include<vector>
#include<queue>
#include<unordered_map>
#include<cstdint>
#include<string>
#include<optional>
#include<map>

class BPETokenizer
{
      private:

      class Node
      {
           public:
           uint32_t id;
           Node* next;
           Node* prev;


           Node()
           {
                this->id=0;
                this->next=nullptr;
                this->prev=nullptr;
           }

           Node(uint32_t id)
           {
                this->id=id;
                this->next=nullptr;
                this->prev=nullptr;
           }
      };

      long long tokenPairCount=0;

      std::unordered_map<std::string,uint32_t> vocabulary;
      std::unordered_map<uint32_t,std::string> decoderMap;

      std::unordered_map<std::string,std::vector<uint32_t>>tokenSeqMap;
      std::unordered_map<std::string,long long> tokenFreqMap;

      std::unordered_map<uint64_t,uint32_t> mergeRules;

      std::vector<std::string> preTokenization(const std::string_view &data);

      void buildTokenIDSeq(const std::vector<std::string>& tokens);

      void updateTokenIDSeq(const std::pair<uint32_t,uint32_t>& targetPair
           ,const uint32_t updateToken);

      std::optional<std::pair<uint32_t,uint32_t>> find_Max_Freq_TokenPair();

      void training();

      std::optional<std::vector<uint32_t>> mergeTokenIds(const std::vector<uint32_t>& tokenIds);

      public:

      void init(const std::string& data);

      void display();

      std::vector<uint32_t> encode(const std::string& data);

      std::string decode(const std::vector<uint32_t>& tokenIDs);
};
