#pragma once
#include <unordered_set>
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

      //this map stores the individual token to id token->id mapping
      std::unordered_map<std::string,uint32_t> vocabulary;

      //this map stores the id->token mapping, used for decoding the message
      std::unordered_map<uint32_t,std::string> decoderMap;

      //this map stores the token chunk-> individual token list mapping
      //ex low-> l,o,w
      std::unordered_map<std::string,std::vector<uint32_t>>tokenSeqMap;

      //this map stores the freq of every seen token chunk used for weighted counting
      std::unordered_map<std::string,long long> tokenChunkFreqMap;

      //this map stores the freq of individual token pairs
      //ex (l,o)->freq
      std::unordered_map<uint64_t, long long> tokenPairFreqMap;

      //this map is storing the reversed indexing of individual token pairs to the 
      //token chunks where they are seen.
      //(p1,p2)-> set of words where pair p1,p2 occurs
      //ex- low,lower,lowest then
      // (l,o)-> low,lower,lowest
      // (e,s)->lowest ..

      //since unordered_map does not have a built in hash function for std::pair<>
      //we will be using bit masking for the pair keys. our token ids are of uint32_t
      //so we will mask by id1<<32 | id2 into uint64_t key that will be used as the key
      //in this map. We are not using ordered map to prevent the O(logn) complexity of RB tree
      std::unordered_map<uint64_t, std::unordered_set<std::string>>reversedTokenMap;

      //this map contains the merge rules for merging token id1, id2 to some id3. Here also
      //bit masking is used as std::pair<> cannot be used for the unordered_map key
      std::unordered_map<uint64_t,uint32_t> mergeRules;

      std::priority_queue<std::pair<long long,uint64_t>> mergeRankPQ;



      size_t splitPoint(std::string_view data, size_t pos);

      std::vector<std::string> preTokenization(const std::string_view &data);

      void buildTokenIDSeq(const std::vector<std::string>& tokens);

      //this method builds up the reverse indexing token pair-> list of words containing the pair
      //and builds uo the token pair freq map used later for weighted counting
      void construct_ReverseIndexing_And_TokenPair_Freq();

      //this method goes thorigh the token pair freq map and initialises the heap for the intial
      //merge state
      void initialise_MergeRankHeap();

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
