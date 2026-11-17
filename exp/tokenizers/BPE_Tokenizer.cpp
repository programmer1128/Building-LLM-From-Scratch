#include "BPE_Tokenizer.h"
#include <iostream>
#include <queue>
#include<map>
#include <climits>

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

//this method builds up the token Seq Map
void BPETokenizer::buildTokenIDSeq(const std::vector<std::string>&tokens)
{
     for(const std::string &token:tokens)
     {
         //now for every token we build up the byte pair following a sliding window
         //of size=2
         if(tokenFreqMap.find(token)==tokenFreqMap.end())
         {
             tokenSeqMap[token]=std::vector<uint32_t>();
             for(size_t i=0;i<token.size();i++)
             {
                 tokenSeqMap[token].push_back((uint32_t)static_cast<uint8_t>(token[i]));
                 tokenPairCount++;
             }
             tokenFreqMap[token]=0;
         }
         tokenFreqMap[token]++;
     }
}

void BPETokenizer::updateTokenIDSeq(const std::pair<uint32_t,uint32_t>& targetPair,
         const uint32_t updateToken)
{
     //update all the token pairs like the target pairs with the updated token
     //the changes will be made in all the token keys of the tokenSeqMap
     for(auto &key:tokenSeqMap)
     {
         //now for every token key, we iterate the vector of tokenIDs and we make the
         //update to the token ids
         std::vector<uint32_t> updatedIDs;
         std::vector<uint32_t>& seqIDs=key.second;

         if(seqIDs.size()<2)
         {
             continue;
         }

         for(size_t i=0;i<seqIDs.size();i++)
         {
             //if we find a pair that matches our target pair, we simply put the new
             //updated token in place of it, in the new vector and skip the 2 values
             if(i+1<seqIDs.size()&&seqIDs[i]==targetPair.first&&seqIDs[i+1]==targetPair.second)
             {
                 updatedIDs.push_back(updateToken);
                 i++;
                 continue;
             }
             updatedIDs.push_back(seqIDs[i]);
         }
         //after the updation, we make the map point to the updated new ids
         std::swap(seqIDs,updatedIDs);
     }
}

std::optional<std::pair<uint32_t,uint32_t>> BPETokenizer::find_Max_Freq_TokenPair()
{
     long long maxFreq=0;
     std::map<std::pair<uint32_t,uint32_t>,long long> freqMap;
     std::pair<uint32_t,uint32_t> maxFreqPair;
     for(const auto& key:tokenSeqMap)
     {
         const std::string &token=key.first;
         const std::vector<uint32_t>& seqIDs=key.second;
         const long long tokenFreq=tokenFreqMap[token];
         for(size_t i=0;i+1<seqIDs.size();i++)
         {
             std::pair<uint32_t,uint32_t>p=std::make_pair(seqIDs[i],seqIDs[i+1]);

             if(freqMap.find(p)==freqMap.end())
             {
                 freqMap[p]=0;
             }
             freqMap[p]+=tokenFreq;

         }
     }

     for(const auto& key:freqMap)
     {
         //if freq is more than max freq then we just replace it.
         std::pair<uint32_t,uint32_t> p=key.first;
         if(maxFreq<freqMap[p])
         {
             maxFreq=freqMap[p];
             maxFreqPair=p;
         }
         else if(maxFreq==freqMap[p])
         {
             //in case of equality we keep the pair with the lower lexicographical pair
             std::string s1=decoderMap[p.first];
             std::string s2=decoderMap[p.second];

             std::string p1=s1+s2;

             std::string s3=decoderMap[maxFreqPair.first];
             std::string s4=decoderMap[maxFreqPair.second];

             std::string p2=s3+s4;

             //now we compare p1 and p2 lexicographically and the smaller one is kept
             if(p1<p2)
             {
                 maxFreqPair=p;
             }
         }
     }


     if(maxFreq<=1)
     {
         return std::nullopt;
     }
     return maxFreqPair;
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
         //display();
     }
}


void BPETokenizer::init(const std::string &data)
{
     vocabulary.clear();
     decoderMap.clear();
     tokenFreqMap.clear();
     tokenSeqMap.clear();
     mergeRules.clear();
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

     std::vector<std::string> tokens=preTokenization(data);
     buildTokenIDSeq(tokens);
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
         std::cout<<key.first<<" "<<tokenFreqMap[key.first]<<" ";
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
         if(tokenFreqMap.find(token)==tokenFreqMap.end())
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
             tokenFreqMap[token]=0;
         }
         tokenFreqMap[token]++;
     }
}




void BPETokenizer::display()
{
     for(const auto &key:tokenSeqMap)
     {
         std::cout<<key.first<<" "<<tokenFreqMap[key.first]<<" ";
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
