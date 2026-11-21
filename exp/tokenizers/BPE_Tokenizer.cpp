#include "BPE_Tokenizer.h"
#include <iostream>
#include <optional>
#include <queue>
#include <unordered_map>

std::vector<std::string> BPETokenizer::preTokenization(const std::string_view &data)
{
     std::vector<std::string> tokens;
     std::string token="";
     //extract the first word before the first space
     for(size_t i=0;i<data.size();i++)
     {
         if(data[i]==' ')
         {
             //a new token with leading space has been built, so we
             //push this into the tokens vector
             if(!token.empty())
             {
                 tokens.push_back(token);
             }
             token="";
         }
         token+=data[i];
     }
     //push the last word incase there is no space at the end of the string
     if(!token.empty())
     {
         tokens.push_back(token);
     }

     return tokens;
}

//to be changed for optimisation
//this method builds up the token Seq Map
void BPETokenizer::buildTokenIDSeq(const std::vector<std::string>&tokens)
{
     //construct the individual token seq for each of the individual token chunks
     for(const std::string &token:tokens)
     {
         //for every token build up the token seq map
         //ex= low-> l,o,w
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
                     reversedTokenMap[key]=std::unordered_set<std::string>();
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


//to be changed for optimisation
void BPETokenizer::updateTokenIDSeq(const std::pair<uint32_t, uint32_t>& targetPair,
                                    const uint32_t updateToken)
{
     uint64_t targetKey = (static_cast<uint64_t>(targetPair.first) << 32) | targetPair.second;

     // Move out or copy, then drop the dead inverted-index bucket for targetKey
     std::unordered_set<std::string> chunksToUpdate = std::move(reversedTokenMap[targetKey]);
     reversedTokenMap.erase(targetKey);

     // Tracks net frequency change for every affected pair across this merge round
     std::unordered_map<uint64_t, long long> pairDelta;

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

//to be changed for optimisation
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

void BPETokenizer::training()
{
     //the training method does the pretraining of the tokenizer, where it learns how to
     //perform the merges of the token ids, into a new single token

     //the training algorithm finds the pair occuring the most number of times, from our
     //token seq map, then we replace that with a new tokenID as the concatenation of the 2 IDs
     //if 'l', 'o' pair is the most occuring, we concatenate them to 'lo' and it becomes a new
     //ID, like this the training continues till we have the number of tokens as 1

     //for n tokens available, we can have at max n-1 token pairs
     tokenPairCount=tokenPairCount-1;

     uint32_t tokenID=256;

     while(true)
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

         //put new token in vocabulary and decoder map
         vocabulary[newToken]=tokenID;
         decoderMap[tokenID]=newToken;

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


void BPETokenizer::init(const std::string &data)
{
     vocabulary.clear();
     decoderMap.clear();
     tokenChunkFreqMap.clear();
     tokenSeqMap.clear();
     mergeRules.clear();
     tokenPairFreqMap.clear();
     reversedTokenMap.clear();
     mergeRankPQ = {};
     tokenPairCount = 0;
     //initilise the vocabulary and the decoder map with teh first 255 characters
     uint32_t id=0;
     for(int i=0;i<256;i++)
     {
         std::string token(1, static_cast<char>(i));
         vocabulary[token]=id;
         decoderMap[id]=token;
         //std::cout<<decoderMap[id]<<"\n";
         id++;
     }

     //break the data into individual tokens
     std::vector<std::string> tokens=preTokenization(data);

     //construct the token id seq
     buildTokenIDSeq(tokens);

     //construct the reverse indexing and the token pair freq map
     construct_ReverseIndexing_And_TokenPair_Freq();

     //after the token Pair Freq is calculated we need to push these intial token pairs
     //into the priroity queue mergeRankPQ
     initialise_MergeRankHeap();

     //display();
     training();
}

std::optional<std::vector<uint32_t>> BPETokenizer::mergeTokenIds(const std::vector<uint32_t>& tokenIds)
{
     std::vector<uint32_t> mergedTokenIDs;
     std::optional<std::pair<uint32_t,uint32_t>> mergePair=std::nullopt;
     uint32_t rank= UINT32_MAX;
     for(size_t i=0;i<tokenIds.size();i++)
     {
         if(i+1<tokenIds.size())
         {
             std::pair<uint32_t,uint32_t>p=std::make_pair(tokenIds[i],tokenIds[i+1]);

             uint64_t pairKey= static_cast<uint64_t>(tokenIds[i])<<32 | tokenIds[i+1];

             if(mergeRules.find(pairKey)!=mergeRules.end())
             {
                 uint32_t tokenPairRank=mergeRules[pairKey]-256;
                 if(rank>tokenPairRank)
                 {
                     mergePair=p;
                     rank=tokenPairRank;
                 }
             }
         }
     }
     //if not a single pair in the current vector has a rule, then we return the nullopt
     if(!mergePair.has_value())
     {
         return std::nullopt;
     }

     //if there is a valid mergePair, we update the vector
     std::pair<uint32_t,uint32_t> validMergePair=*mergePair;
     uint32_t first=validMergePair.first;
     uint32_t sec=validMergePair.second;
     for(size_t i=0;i<tokenIds.size();i++)
     {
         if(i+1<tokenIds.size()&&tokenIds[i]==first&&tokenIds[i+1]==sec)
         {
             //valid pair, so we merge them
             uint64_t pairKey= static_cast<uint64_t>(first)<<32 | sec;
             mergedTokenIDs.push_back(mergeRules[pairKey]);
             i++;
             continue;
         }
         //if not a valid pair, then we just push the current id
         mergedTokenIDs.push_back(tokenIds[i]);
     }
     return mergedTokenIDs;
}

std::vector<uint32_t> BPETokenizer::encode(const std::string& data)
{
     //get the broken down tokens using the same preTokenization method
     std::vector<std::string> tokens= preTokenization(data);

     std::unordered_map<std::string,std::vector<uint32_t>> tokenVectorMap;

     //vector for the merged token ids
     std::vector<uint32_t> tokenIDs;

     for(const std::string& token:tokens)
     {
         if(tokenVectorMap.find(token)==tokenVectorMap.end())
         {
             tokenVectorMap[token]=std::vector<uint32_t>();
             for(size_t i=0;i<token.size();i++)
             {
                 tokenVectorMap[token].push_back((uint32_t)static_cast<uint8_t>(token[i]));
             }
         }
     }

     for(auto & key:tokenVectorMap)
     {
         //now for every set of vector, we continue to merge token ids in it, till there
         //is no rule available for the set of token ids
         std::vector<uint32_t> mergedTokens(key.second);
         while(true)
         {
             std::optional<std::vector<uint32_t>> mergedTokensOptional=mergeTokenIds(mergedTokens);
             if(!mergedTokensOptional.has_value())
             {
                 //no rule condition
                 break;
             }
             //if there was a rule, and merge was performed, then we replace the merged token vector
             mergedTokens=*mergedTokensOptional;
         }
         //append the new set of merged tokens into the tokenID vector
         std::swap(key.second,mergedTokens);
     }

     for(const std::string &token:tokens)
     {
         tokenIDs.append_range(tokenVectorMap.at(token));
     }

     return tokenIDs;
}


std::string BPETokenizer::decode(const std::vector<uint32_t>& tokenIDs)
{
     std::string decodedText="";
     for(const uint32_t &tokenId:tokenIDs)
     {
         if(decoderMap.find(tokenId)!=decoderMap.end())
         {
             decodedText+=decoderMap[tokenId];
         }
     }
     return decodedText;
}


void BPETokenizer::display()
{
     for(const auto &key:tokenSeqMap)
     {
         std::cout<<key.first<<" "<<tokenChunkFreqMap[key.first]<<" ";
         for(const uint32_t& id:key.second)
         {
             std::cout<<id<<" ";
         }
         std::cout<<"\n";
     }
}


/*

//this method builds up the token Seq Map
void BPETokenizer::buildTokenIDSeq(const std::vector<std::string>&tokens)
{
     for(const std::string &token:tokens)
     {
         //now for every token we build up the byte pair following a sliding window
         //of size=2
         if(tokenChunkFreqMap.find(token)==tokenChunkFreqMap.end())
         {
             Node* newSeqHead= nullptr;
             Node* temp=nullptr;
             for(size_t i=0;i<token.size();i++)
             {
                 //push the token ids
                 if(newSeqHead==nullptr)
                 {
                     newSeqHead= new Node((uint32_t)(token[i]));
                     temp=newSeqHead;
                 }
                 else
                 {
                     Node* newNode= new Node((uint32_t)(token[i]));
                     temp->next=newNode;
                     newNode->prev=temp;
                     temp=temp->next;
                 }
                 tokenPairCount++;
             }
             tokenSeqMap[token]=newSeqHead;
            //  std::cout<<"head id"<<newSeqHead->id<<" "<<"\n";
            //  std::cout<<"next id"<<temp->id<<" "<<"\n";
             //display();
             tokenChunkFreqMap[token]=0;
         }
         tokenChunkFreqMap[token]++;
     }
}




void BPETokenizer::display()
{
     for(const auto &key:tokenSeqMap)
     {
         std::cout<<key.first<<" "<<tokenChunkFreqMap[key.first]<<" ";
         Node* temp=key.second;
         int p=0;
         while(temp!=nullptr)
         {
             std::cout<<temp->id<<" ";
             temp=temp->next;
             p++;
         }
         std::cout<<"\n";
     }
}*/
