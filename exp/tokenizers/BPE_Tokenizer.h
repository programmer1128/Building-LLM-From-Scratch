#pragma once
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <array>
#include <vector>
#include <queue>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <optional>

class BPETokenizer
{
      private:

      long long tokenPairCount=0;

      struct TransparentStringHash
      {
           using is_transparent = void;

           size_t operator()(std::string_view sv) const noexcept
           {
                return boost::hash<std::string_view>{}(sv);
           }
      };

      //this map stores the individual token to id token->id mapping
      boost::unordered_flat_map<std::string, uint32_t, TransparentStringHash, std::equal_to<>> vocabulary;
      
      //this map stores the id->token mapping, used for decoding the message
      std::vector<std::string> decoderMap;

      //this map stores the token chunk-> individual token list mapping
      //ex low-> l,o,w
      boost::unordered_flat_map<std::string,std::vector<uint32_t>> tokenSeqMap;

      //this map stores the freq of every seen token chunk used for weighted counting
      boost::unordered_flat_map<std::string,long long> tokenChunkFreqMap;

      //this map stores the freq of individual token pairs
      //ex (l,o)->freq
      boost::unordered_flat_map<uint64_t, long long> tokenPairFreqMap;

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
      boost::unordered_flat_map<uint64_t, boost::unordered_flat_set<std::string>> reversedTokenMap;

      //this map contains the merge rules for merging token id1, id2 to some id3. Here also
      //bit masking is used as std::pair<> cannot be used for the unordered_map key
      boost::unordered_flat_map<uint64_t,uint32_t> mergeRules;

      //special tokens (e.g., <|endoftext|>, <|im_start|>) that bypass BPE splitting
      boost::unordered_flat_map<std::string, uint32_t, TransparentStringHash, std::equal_to<>> specialTokens;
      boost::unordered_flat_set<uint32_t> specialTokenIds;
      std::vector<std::pair<std::string, uint32_t>> specialTokensByLen;
      std::array<bool, 256> specialFirstByte{};

      std::priority_queue<std::pair<long long,uint64_t>> mergeRankPQ;

      //used in merging during the encoding section
      struct Part
      {
           uint32_t id;
           uint32_t mergeId;
      };

      std::vector<std::string_view> preTokenization(std::string_view data) const;

      void buildTokenIDSeq(const std::vector<std::string_view>& tokens);

      //this method builds up the reverse indexing token pair-> list of words containing the pair
      //and builds uo the token pair freq map used later for weighted counting
      void construct_ReverseIndexing_And_TokenPair_Freq();

      //this method goes thorigh the token pair freq map and initialises the heap for the intial
      //merge state
      void initialise_MergeRankHeap();

      void updateTokenIDSeq(const std::pair<uint32_t,uint32_t>& targetPair
           ,const uint32_t updateToken);

      std::optional<std::pair<uint32_t,uint32_t>> find_Max_Freq_TokenPair();

      void training(size_t maxVocabSize);

      void mergeTokenIds(std::string_view data, std::vector<uint32_t>& outputTokenIDs) const;

      void encodeOrdinaryInto(std::string_view data, std::vector<uint32_t>& outputTokenIDs) const;

      uint32_t lookupMerge(uint32_t leftId,uint32_t rightId) const;

      void rebuildSpecialTokenIndex();

      uint32_t addSpecialToken(const std::string& token);

      void addSpecialTokens(const std::vector<std::string>& tokens);

      std::optional<uint32_t> getSpecialTokenId(std::string_view token) const;

      size_t getVocabSize() const noexcept { return decoderMap.size(); }

      std::vector<uint32_t> encodeOrdinary(std::string_view data) const;

      public:

      void init(const std::string& data, size_t maxVocabSize = std::numeric_limits<size_t>::max());

      void save(const std::string& filePath) const;

      void load(const std::string& filePath);

      void display() const;

      std::vector<uint32_t> encode(std::string_view data, bool allowSpecial = true) const;

      std::string decode(const std::vector<uint32_t>& tokenIDs, bool skipSpecialTokens = false) const;
};
