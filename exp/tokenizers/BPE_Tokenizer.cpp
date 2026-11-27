#include "BPE_Tokenizer.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <stdexcept>
#include "SplitStateMachine.h"

std::vector<std::string_view> BPETokenizer::preTokenization(std::string_view data) const
{
     return SplitStateMachine::split(data);
}


//this method builds up the token Seq Map
void BPETokenizer::buildTokenIDSeq(const std::vector<std::string_view>& tokens)
{
     //construct the individual token seq for each of the individual token chunks
     for(std::string_view tokenView : tokens)
     {
         //for every token build up the token seq map
         //ex= low-> l,o,w
         std::string token(tokenView);
         if(tokenChunkFreqMap.find(token)==tokenChunkFreqMap.end())
         {
             tokenSeqMap[token]=std::vector<uint32_t>();
             for(size_t i=0;i<token.size();i++)
             {
                 tokenSeqMap[token].push_back((uint32_t)static_cast<uint8_t>(token[i]));
             }
             tokenChunkFreqMap[token]=0;
         }
         //count the freq of this token that will be used later in weighted freq counting
         tokenChunkFreqMap[token]++;
     }
}

//this method builds up the reverse token indexing tokenPair-> list of words having the pair
//and builds up the tokenPairFreq Map that is later used in code for weighted freq counting
void BPETokenizer::construct_ReverseIndexing_And_TokenPair_Freq()
{
     for(const auto & tokenKeys:tokenChunkFreqMap)
     {
         const std::string token=tokenKeys.first;
         const std::vector<uint32_t>& tokenIds=tokenSeqMap[token];
         const long long tokenChunkFreq= tokenChunkFreqMap[token];

         for(size_t i=0; i<tokenIds.size();i++)
         {
             //build the reversed indexing of individual token pairs to the token chunks
             if(i+1<tokenIds.size())
             {
                 //constructing bit masked key
                 uint32_t id1= tokenIds[i];
                 uint32_t id2= tokenIds[i+1];
 
                 uint64_t key= static_cast<uint64_t>(id1)<<32 | id2;

                 //insert the pair into the reversed indexed map
                 if(reversedTokenMap.find(key)==reversedTokenMap.end())
                 {
                     reversedTokenMap[key]=boost::unordered_flat_set<std::string>();
                 }
                 reversedTokenMap[key].insert(token);
 
                 //using the same key, we insert in the individual token pair freq map
                 //as well, which will be later used for optimised merging in the priority
                 //queue for the training
                 if(tokenPairFreqMap.find(key)==tokenPairFreqMap.end())
                 {
                     tokenPairFreqMap[key]=0;
                 }
                 //weighted freq counting
                 tokenPairFreqMap[key]+=tokenChunkFreq;
             }
             tokenPairCount++;
         }
     }
}

//this method initialises the max heap with the token pairs and the freq for the intial merge state
void BPETokenizer::initialise_MergeRankHeap()
{
     //go through the token pairs in the tokenPairFreq map and push them into the heap
     //for the intitial merge state
     for(const auto& tokenPairKeys:tokenPairFreqMap)
     {
         const uint64_t key=tokenPairKeys.first;
         const long long freq=tokenPairKeys.second;

         if(freq<=1)
         {
             continue;
         }
         mergeRankPQ.push(std::make_pair(freq,key));
     }
}


//this method updates the token id sequences by merging the max freq token pairs
//into the new token
void BPETokenizer::updateTokenIDSeq(const std::pair<uint32_t, uint32_t>& targetPair,
                                    const uint32_t updateToken)
{
     uint64_t targetKey = (static_cast<uint64_t>(targetPair.first) << 32) | targetPair.second;

     // Move out or copy, then drop the dead inverted-index bucket for targetKey
     boost::unordered_flat_set<std::string> chunksToUpdate = std::move(reversedTokenMap[targetKey]);
     reversedTokenMap.erase(targetKey);

     // Tracks net frequency change for every affected pair across this merge round
     boost::unordered_flat_map<uint64_t, long long> pairDelta;

     for (const std::string& tokenChunk : chunksToUpdate)
     {
         std::vector<uint32_t>& seqIDs = tokenSeqMap[tokenChunk];
         if (seqIDs.size() < 2) 
         {
             continue;
         }
         std::vector<uint32_t> updatedIDs;
         updatedIDs.reserve(seqIDs.size());

         for (size_t i = 0; i < seqIDs.size(); i++)
         {
             if (i+1<seqIDs.size() && seqIDs[i] == targetPair.first && seqIDs[i + 1] == targetPair.second)
             {
                 updatedIDs.push_back(updateToken);
                 i++;
                 continue;
             } 
            updatedIDs.push_back(seqIDs[i]);
         }

         // If no merge occurred (stale entry in reversedTokenMap), skip
         if (updatedIDs.size() == seqIDs.size()) 
         {
             continue;
         }

         long long chunkFreq = tokenChunkFreqMap[tokenChunk];

         //subtract weighted freq of all pairs from the old sequence
         for (size_t j = 0; j + 1 < seqIDs.size(); j++)
         {
             uint64_t oldKey = (static_cast<uint64_t>(seqIDs[j]) << 32) | seqIDs[j + 1];
             pairDelta[oldKey] -= chunkFreq;
         }

         std::swap(seqIDs, updatedIDs);

         //add all pairs from the new sequence and update inverted index. Here seqIDS have been
         //swapped with the new updated ids, so we can use the same seq ids vector only
         for (size_t j = 0; j + 1 < seqIDs.size(); j++)
         {
             uint64_t newKey = (static_cast<uint64_t>(seqIDs[j]) << 32) | seqIDs[j + 1];
             pairDelta[newKey] += chunkFreq;
             reversedTokenMap[newKey].insert(tokenChunk);
         }
     }

    //Apply net deltas and batch-push changed pairs to the priority queue
     for (const auto& [pairKey, delta] : pairDelta)
     {
         if (delta == 0) 
         {
             continue; // Untouched pair in the chunk; skip to avoid heap bloat
         }

         long long newFreq = (tokenPairFreqMap[pairKey] += delta);

         if (newFreq <= 0)
         {
             tokenPairFreqMap.erase(pairKey);
         }
         else if (newFreq > 1)
         {
             mergeRankPQ.push({newFreq, pairKey});
         }
     }
}

//finds the max freq token pair required in BPE by using alzy deletion of the priority queue
std::optional<std::pair<uint32_t,uint32_t>> BPETokenizer::find_Max_Freq_TokenPair()
{
     //here we have to get the max freq pair from the rank priority queue. Now the queue
     //may hold stale entries, so first we have to verify the popping of the queue and
     //the value of freq in our global freq map
     while(!mergeRankPQ.empty())
     {
         auto topNode=mergeRankPQ.top();
         long long heapFreq= topNode.first;
         auto it = tokenPairFreqMap.find(topNode.second);
         mergeRankPQ.pop();

         if(it==tokenPairFreqMap.end()||heapFreq!=it->second)
         {
             //this is a stale entry
             continue;
         }

         if(heapFreq<=1)
         {
             return std::nullopt;
         }
         //the pairs have been bit masked to 64 bit uint64_t, so we have to 
         //unpack them and then push them into the priority queue
         //the first id is bit masked into the first 32 positions by bit shift left <<32
         //and the second is or into the right next 32 positions
       
         //unmasking the ids
         uint64_t pairKey= topNode.second;
         uint32_t id1 = pairKey >> 32;
         uint32_t id2 = pairKey & 0xFFFFFFFF;
         return std::make_pair(id1, id2);
     }
     return std::nullopt;
}

void BPETokenizer::training(size_t maxVocabSize)
{
     //the training method does the pretraining of the tokenizer, where it learns how to
     //perform the merges of the token ids, into a new single token

     //the training algorithm finds the pair occuring the most number of times, from our
     //token seq map, then we replace that with a new tokenID as the concatenation of the 2 IDs
     //if 'l', 'o' pair is the most occuring, we concatenate them to 'lo' and it becomes a new
     //ID, like this the training continues till we reach maxVocabSize or no pairs repeat

     //for n tokens available, we can have at max n-1 token pairs
     tokenPairCount=tokenPairCount-1;

     uint32_t tokenID=256;

     while(tokenID < maxVocabSize)
     {
         //get the tokenID pair with the max freq
         std::optional<std::pair<uint32_t,uint32_t>> maxFreqPairOptional=find_Max_Freq_TokenPair();

         if(!maxFreqPairOptional.has_value())
         {
             break;
         }
         std::pair<uint32_t,uint32_t> maxFreqPair=*maxFreqPairOptional;
         //get the individual tokens for the maxFreqPair token IDs
         std::string s1=decoderMap[maxFreqPair.first];
         std::string s2=decoderMap[maxFreqPair.second];

         //construct new token
         std::string newToken=s1+s2;

         //put new token in vocabulary (preserving lowest-rank ID on duplicate string) and decoder map
         vocabulary.try_emplace(newToken, tokenID);
         decoderMap.push_back(newToken);

         //update the mergeRulesMap
         //mergeRules[maxFreqPair]=tokenID;
         //build a bit masked key for storing both values
         uint64_t pairKey= (static_cast<uint64_t>(maxFreqPair.first)<<32)
             | maxFreqPair.second;

         mergeRules[pairKey]=tokenID;

         //update the token seq Map to replace the old maxFreq token pairs with new tokenID
         updateTokenIDSeq(maxFreqPair,tokenID);
         tokenID++;
         //std::cout<<"hello\n";
         //display();
     }
}


void BPETokenizer::init(const std::string &data, size_t maxVocabSize)
{
     if (maxVocabSize < 256)
     {
         throw std::invalid_argument("maxVocabSize must be at least 256 for byte-level BPE");
     }

     hasImportedRanks = false;
     vocabulary.clear();
     decoderMap.clear();
     tokenChunkFreqMap.clear();
     tokenSeqMap.clear();
     mergeRules.clear();
     tokenPairFreqMap.clear();
     reversedTokenMap.clear();
     specialTokens.clear();
     specialTokenIds.clear();
     specialTokensByLen.clear();
     specialFirstByte.fill(false);
     mergeRankPQ = {};
     tokenPairCount = 0;
     //initilise the vocabulary and the decoder map with the first 256 byte values
     uint32_t id=0;
     decoderMap.resize(256);
     
     for(int i=0;i<256;i++)
     {
         std::string token(1, static_cast<char>(i));
         vocabulary.try_emplace(token, id);
         decoderMap[id]=token;
         //std::cout<<decoderMap[id]<<"\n";
         id++;
     }

     //break the data into individual tokens
     std::vector<std::string_view> tokens=preTokenization(data);

     //construct the token id seq
     buildTokenIDSeq(tokens);

     //construct the reverse indexing and the token pair freq map
     construct_ReverseIndexing_And_TokenPair_Freq();

     //after the token Pair Freq is calculated we need to push these intial token pairs
     //into the priroity queue mergeRankPQ
     initialise_MergeRankHeap();

     //display();
     training(maxVocabSize);
}


uint32_t BPETokenizer::lookupMerge(uint32_t leftId, uint32_t rightId) const
{
     uint64_t pairKey = (static_cast<uint64_t>(leftId) << 32) | rightId;
     auto it = mergeRules.find(pairKey);
     return (it != mergeRules.end()) ? it->second : UINT32_MAX;
}

void BPETokenizer::mergeTokenIds(std::string_view chunk, std::vector<uint32_t>& outputTokenIDs) const
{
     if(chunk.empty())
     {
         return;
     }
     thread_local std::vector<Part> parts;
     parts.clear();
     const size_t n= chunk.size();
     parts.resize(n);

     for(size_t i=0;i<chunk.size();i++)
     {
         parts[i].id=static_cast<uint32_t>(static_cast<uint8_t>(chunk[i]));
     }

     for(size_t i=0;i+1<n;i++)
     {
         parts[i].mergeId = lookupMerge(parts[i].id, parts[i + 1].id);
     }
     parts[n-1].mergeId=UINT32_MAX;

     //now we keep on merging till there are no valid rules to merge
     //since parts have id and merge id for index and after merging the next
     //index drops out, it will continue to merge till there is no merge rule existing
     //for the tokens available
     while(true)
     {
         uint32_t minMergedId = UINT32_MAX;
         size_t minIdx = 0;

         // Pure integer scan in L1 cache (strict '<' preserves leftmost tie-break)
         for (size_t i = 0;i+1< parts.size(); ++i)
         {
             if (parts[i].mergeId < minMergedId)
             {
                 minMergedId=parts[i].mergeId;
                 minIdx = i;
             }
         }

         if (minMergedId == UINT32_MAX)
         {
             break;
         }

         //merge in-place and drop the consumed right neighbor
         parts[minIdx].id = minMergedId;
         parts.erase(parts.begin() + minIdx + 1);

         //recompute only the 2 affected neighbor pairs (at most 2 hash lookups)
         if (minIdx > 0)
         {
             parts[minIdx - 1].mergeId=lookupMerge(parts[minIdx - 1].id, parts[minIdx].id);
         }
         if (minIdx + 1 < parts.size())
         {
             parts[minIdx].mergeId=lookupMerge(parts[minIdx].id, parts[minIdx + 1].id);
         }
         else
         {
             parts[minIdx].mergeId = UINT32_MAX;
         }
     }

     //push the final token ids to the output token vector
     for(const Part& p:parts)
     {
         outputTokenIDs.push_back(p.id);
     }
}

void BPETokenizer::rebuildSpecialTokenIndex()
{
     specialTokenIds.clear();
     specialTokensByLen.clear();
     specialFirstByte.fill(false);

     specialTokenIds.reserve(specialTokens.size());
     specialTokensByLen.reserve(specialTokens.size());

     for (const auto& [tokenStr, id] : specialTokens)
     {
         specialTokenIds.insert(id);
         specialTokensByLen.emplace_back(tokenStr, id);
         if (!tokenStr.empty())
         {
             specialFirstByte[static_cast<uint8_t>(tokenStr[0])] = true;
         }
     }

     // Sort longest-first so prefix overlaps (e.g., "<|end|>" vs "<|endoftext|>") match greedily
     std::sort(specialTokensByLen.begin(), specialTokensByLen.end(),
               [](const auto& a, const auto& b) {
                   if (a.first.size() != b.first.size())
                   {
                       return a.first.size() > b.first.size();
                   }
                   return a.second < b.second;
               });
}

uint32_t BPETokenizer::addSpecialToken(const std::string& token)
{
     if (decoderMap.size() < 256)
     {
         throw std::logic_error("Initialize or load tokenizer before adding special tokens");
     }
     if (token.empty())
     {
         throw std::invalid_argument("Special token cannot be empty");
     }

     auto it = specialTokens.find(token);
     if (it != specialTokens.end())
     {
         return it->second;
     }

     uint32_t newId = static_cast<uint32_t>(decoderMap.size());
     decoderMap.push_back(token);
     specialTokens.emplace(token, newId);
     rebuildSpecialTokenIndex();
     return newId;
}

void BPETokenizer::addSpecialTokens(const std::vector<std::string>& tokens)
{
     if (decoderMap.size() < 256)
     {
         throw std::logic_error("Initialize or load tokenizer before adding special tokens");
     }

     bool addedAny = false;
     for (const std::string& token : tokens)
     {
         if (token.empty())
         {
             throw std::invalid_argument("Special token cannot be empty");
         }
         if (specialTokens.find(token) != specialTokens.end())
         {
             continue;
         }
         uint32_t newId = static_cast<uint32_t>(decoderMap.size());
         decoderMap.push_back(token);
         specialTokens.emplace(token, newId);
         addedAny = true;
     }

     if (addedAny)
     {
         rebuildSpecialTokenIndex();
     }
}

std::optional<uint32_t> BPETokenizer::getSpecialTokenId(std::string_view token) const
{
     auto it = specialTokens.find(token);
     if (it == specialTokens.end())
     {
         return std::nullopt;
     }
     return it->second;
}

void BPETokenizer::encodeOrdinaryInto(std::string_view data, std::vector<uint32_t>& outputTokenIDs) const
{
     if (data.empty())
     {
         return;
     }

     std::vector<std::string_view> tokens = preTokenization(data);

     for (const std::string_view vt : tokens)
     {
         if (vt.size() == 1)
         {
             outputTokenIDs.push_back(static_cast<uint32_t>(static_cast<uint8_t>(vt[0])));
             continue;
         }
         auto it = vocabulary.find(vt);

         if (it != vocabulary.end())
         {
             outputTokenIDs.push_back(it->second);
             continue;
         }

         mergeTokenIds(vt, outputTokenIDs);
     }
}

std::vector<uint32_t> BPETokenizer::encodeOrdinary(std::string_view data) const
{
     std::vector<uint32_t> outputTokenIDs;
     outputTokenIDs.reserve(data.size() / 3);
     encodeOrdinaryInto(data, outputTokenIDs);
     return outputTokenIDs;
}

std::vector<uint32_t> BPETokenizer::encode(std::string_view data, bool allowSpecial) const
{
     if (!allowSpecial || specialTokensByLen.empty())
     {
         return encodeOrdinary(data);
     }

     std::vector<uint32_t> outputTokenIDs;
     outputTokenIDs.reserve(data.size() / 3);

     size_t ordinaryStart = 0;
     size_t scan = 0;

     while (scan < data.size())
     {
         if (specialFirstByte[static_cast<uint8_t>(data[scan])])
         {
             std::string_view remaining = data.substr(scan);
             bool matched = false;

             for (const auto& [specialStr, specialId] : specialTokensByLen)
             {
                 if (remaining.starts_with(specialStr))
                 {
                     if (scan > ordinaryStart)
                     {
                         encodeOrdinaryInto(data.substr(ordinaryStart, scan - ordinaryStart), outputTokenIDs);
                     }
                     outputTokenIDs.push_back(specialId);
                     scan += specialStr.size();
                     ordinaryStart = scan;
                     matched = true;
                     break;
                 }
             }

             if (matched)
             {
                 continue;
             }
         }
         ++scan;
     }

     if (ordinaryStart < data.size())
     {
         encodeOrdinaryInto(data.substr(ordinaryStart), outputTokenIDs);
     }

     return outputTokenIDs;
}


std::string BPETokenizer::decode(const std::vector<uint32_t>& tokenIDs, bool skipSpecialTokens) const
{
     size_t totalBytes = 0;
     for (uint32_t id : tokenIDs)
     {
         if (id >= decoderMap.size())
         {
             throw std::out_of_range("Token ID out of vocabulary range during decode");
         }
         if (skipSpecialTokens && specialTokenIds.contains(id))
         {
             continue;
         }
         totalBytes += decoderMap[id].size();
     }

     //single exact heap allocation
     std::string decodedText;
     decodedText.reserve(totalBytes);

     //sequential copy into pre-allocated buffer
     for (uint32_t id : tokenIDs)
     {
         if (skipSpecialTokens && specialTokenIds.contains(id))
         {
             continue;
         }
         decodedText.append(decoderMap[id]);
     }

     return decodedText;
}


void BPETokenizer::display() const
{
     for(const auto &key:tokenSeqMap)
     {
         std::cout<<key.first<<" "<<tokenChunkFreqMap.at(key.first)<<" ";
         for(const uint32_t& id:key.second)
         {
             std::cout<<id<<" ";
         }
         std::cout<<"\n";
     }
}


namespace
{
    constexpr char kMagic[4] = {'A', 'B', 'P', 'E'};
    constexpr uint32_t kFormatVersion = 1;
    constexpr uint32_t kMaxSerializedMerges = 10'000'000;
    constexpr uint32_t kMaxSerializedSpecials = 1'000'000;
    constexpr uint32_t kMaxSpecialTokenLen = 65'536;

    struct DiskMergeRecord
    {
        uint32_t leftId;
        uint32_t rightId;
        uint32_t newId;
    };
    static_assert(sizeof(DiskMergeRecord) == 12, "DiskMergeRecord must be tightly packed");
}

void BPETokenizer::save(const std::string& filePath) const
{
     if (hasImportedRanks)
     {
          throw std::logic_error("ABPE v1 cannot store imported rank vocabularies");
     }

     if (decoderMap.size() < 256)
     {
         throw std::logic_error("Cannot save an uninitialized tokenizer");
     }

     std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
     if (!out)
     {
         throw std::runtime_error("Failed to open file for writing: " + filePath);
     }

     //write file header
     out.write(kMagic, sizeof(kMagic));
     out.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));

     //extract and sort merge rules by newId ascending (preserves rank order)
     std::vector<DiskMergeRecord> sortedMerges;
     sortedMerges.reserve(mergeRules.size());

     for (const auto& [pairKey, newId] : mergeRules)
     {
         uint32_t leftId = static_cast<uint32_t>(pairKey >> 32);
         uint32_t rightId = static_cast<uint32_t>(pairKey & 0xFFFFFFFFULL);
         sortedMerges.push_back({leftId, rightId, newId});
     }

     std::sort(sortedMerges.begin(), sortedMerges.end(),
            [](const DiskMergeRecord& a, const DiskMergeRecord& b) {
                 return a.newId < b.newId;
     });

     uint32_t numMerges = static_cast<uint32_t>(sortedMerges.size());
     out.write(reinterpret_cast<const char*>(&numMerges), sizeof(numMerges));

     if (!sortedMerges.empty())
     {
         out.write(reinterpret_cast<const char*>(sortedMerges.data()),
                  static_cast<std::streamsize>(sortedMerges.size() * sizeof(DiskMergeRecord)));
     }

     //sort special tokens by id ascending for deterministic binary output
     std::vector<std::pair<uint32_t, std::string_view>> sortedSpecials;
     sortedSpecials.reserve(specialTokens.size());
     for (const auto& [tokenStr, id] : specialTokens)
     {
         sortedSpecials.emplace_back(id, tokenStr);
     }
     std::sort(sortedSpecials.begin(), sortedSpecials.end(),
               [](const auto& a, const auto& b) {
                   return a.first < b.first;
               });

     //write special tokens: [count] followed by [id][len][raw bytes]
     uint32_t numSpecials = static_cast<uint32_t>(sortedSpecials.size());
     out.write(reinterpret_cast<const char*>(&numSpecials), sizeof(numSpecials));

     for (const auto& [id, tokenStr] : sortedSpecials)
     {
         uint32_t len = static_cast<uint32_t>(tokenStr.size());
         out.write(reinterpret_cast<const char*>(&id), sizeof(id));
         out.write(reinterpret_cast<const char*>(&len), sizeof(len));
         out.write(tokenStr.data(), static_cast<std::streamsize>(len));
     }

     if (!out.good())
     {
         throw std::runtime_error("I/O error while writing tokenizer binary: " + filePath);
     }
}

void BPETokenizer::load(const std::string& filePath)
{
     std::ifstream in(filePath, std::ios::binary);
     if (!in)
     {
         throw std::runtime_error("Failed to open tokenizer file: " + filePath);
     }

     //validate magic bytes and format version
     char magic[4]{};
     uint32_t version = 0;
     in.read(magic, sizeof(magic));
     in.read(reinterpret_cast<char*>(&version), sizeof(version));

     if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
     {
         throw std::runtime_error("Invalid tokenizer file magic header");
     }
     if (version != kFormatVersion)
     {
         throw std::runtime_error("Unsupported tokenizer binary version");
     }

     //read merge records in bulk (validate stream & bounds BEFORE allocating)
     uint32_t numMerges = 0;
     in.read(reinterpret_cast<char*>(&numMerges), sizeof(numMerges));
     if (!in || numMerges > kMaxSerializedMerges)
     {
         throw std::runtime_error("Corrupted or invalid merge count in tokenizer file: " + filePath);
     }

     std::vector<DiskMergeRecord> sortedMerges(numMerges);
     if (numMerges > 0)
     {
         in.read(reinterpret_cast<char*>(sortedMerges.data()),
                 static_cast<std::streamsize>(numMerges * sizeof(DiskMergeRecord)));
         if (!in)
         {
             throw std::runtime_error("Truncated merge table in tokenizer file: " + filePath);
         }
     }

     hasImportedRanks = false;

     //reset ALL state (including any stale training structures) and rebuild base 0..255 bytes
     mergeRules.clear();
     vocabulary.clear();
     tokenChunkFreqMap.clear();
     tokenSeqMap.clear();
     tokenPairFreqMap.clear();
     reversedTokenMap.clear();
     mergeRankPQ = {};
     tokenPairCount = 0;
     specialTokens.clear();
     specialTokenIds.clear();
     specialTokensByLen.clear();
     specialFirstByte.fill(false);

     decoderMap.assign(256, std::string{});
     mergeRules.reserve(numMerges);
     vocabulary.reserve(256 + numMerges);
     decoderMap.reserve(256 + numMerges);

     for (uint32_t i = 0; i < 256; ++i)
     {
         decoderMap[i] = std::string(1, static_cast<char>(i));
         vocabulary.try_emplace(decoderMap[i], i);
     }

     //replay merges in rank order to populate mergeRules, decoderMap, and vocabulary
     for (uint32_t idx = 0; idx < numMerges; ++idx)
     {
         const DiskMergeRecord& rec = sortedMerges[idx];
         uint32_t expectedId = 256 + idx;
         if (rec.newId != expectedId || rec.leftId >= decoderMap.size() || rec.rightId >= decoderMap.size())
         {
             throw std::runtime_error("Corrupted merge record: invalid or out-of-order token ID");
         }

         uint64_t pairKey = (static_cast<uint64_t>(rec.leftId) << 32) | rec.rightId;
         mergeRules.emplace(pairKey, rec.newId);

         std::string mergedStr = decoderMap[rec.leftId] + decoderMap[rec.rightId];
         decoderMap.push_back(mergedStr);
         vocabulary.try_emplace(std::move(mergedStr), rec.newId);
     }

     //read and register special tokens
     uint32_t numSpecials = 0;
     in.read(reinterpret_cast<char*>(&numSpecials), sizeof(numSpecials));
     if (!in || numSpecials > kMaxSerializedSpecials)
     {
         throw std::runtime_error("Corrupted or invalid special token count in tokenizer file: " + filePath);
     }

     for (uint32_t i = 0; i < numSpecials; ++i)
     {
         uint32_t id = 0;
         uint32_t len = 0;
         in.read(reinterpret_cast<char*>(&id), sizeof(id));
         in.read(reinterpret_cast<char*>(&len), sizeof(len));
         if (!in || len == 0 || len > kMaxSpecialTokenLen)
         {
             throw std::runtime_error("Corrupted special token header in tokenizer file: " + filePath);
         }
         if (id < 256 + numMerges || id >= 256 + numMerges + kMaxSerializedSpecials)
         {
             throw std::runtime_error("Corrupted special token ID out of valid range");
         }

         std::string tokenStr(len, '\0');
         in.read(tokenStr.data(), static_cast<std::streamsize>(len));
         if (!in)
         {
             throw std::runtime_error("Truncated special token payload in tokenizer file: " + filePath);
         }

         if (id >= decoderMap.size())
         {
             decoderMap.resize(id + 1);
         }
         decoderMap[id] = tokenStr;
         specialTokens.emplace(std::move(tokenStr), id);
     }

     rebuildSpecialTokenIndex();
}
