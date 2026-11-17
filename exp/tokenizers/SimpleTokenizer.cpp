#include "SimpleTokenizer.h"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include<set>
#include <ranges>

const std::unordered_set<std::string>delimiterSet={"()","(",")","[]","["
     "]", ",", ".", ":", ";", "?", "_", "!", """", "''", "'", "|", "--",
     "@","#","$","%","^","&","*","+","-","=","/","<",">","<>",">>","<<",
     "{}","{","}"," "
};

const std::string UNK_TOKEN="<unk>";

// //everything between R"( and )" is kept exactly as written
// const std::string_view delimiters = R"([,.:;?_!"()'\\]|--|@#$^&{}+-=/\><)";

std::vector<std::string> Tokenizer::tokenize(const std::string &data)
{
      std::vector<std::string> subTokens;
      std::string subToken = "";
 
      for(int i = 0; i < data.size(); i++)
      {
           char c = data[i];
           // Creates a string of length 1, filled with character 'c'
           std::string extract(1, c); 
           if(c == '(')
           {
                if(i + 1 < data.size() && data[i+1] == ')')
                {
                     extract += data[i+1]; // Just append the char directly
                     i++; // FIX: Skip the next character so you don't process the ')' twice
                }     
           }
           else if(c == '[')
           {
                if(i + 1 < data.size() && data[i+1] == ']')
                {
                     extract += data[i+1];
                     i++;
                }      
           }
           else if(c == '{')
           {
                if(i + 1 < data.size() && data[i+1] == '}')
                {
                     extract += data[i+1];
                     i++;
                }     
           }
          
           if(delimiterSet.find(extract) != delimiterSet.end())
           {
                //  subTokens.insert(extract);
                //  if (!subToken.empty()) 
                //  {
                //       subTokens.insert(subToken);
                //  }
                if(!subToken.empty())
                {
                     subTokens.emplace_back(subToken);
                }
                subTokens.emplace_back(extract);
                subToken = "";
                continue;
           }
           subToken += extract;
      }

      // FIX: If the text doesn't end with a delimiter, 
      // you need to insert the final word that was building up.
      if (!subToken.empty()) 
      {
           subTokens.emplace_back(subToken);
      }

      return subTokens;
}

void Tokenizer::buildVocabulary(const std::string& filePath)
{
      //for every unique token, we build the mapping in our vocabulary
      //and assign an id to it. At the same time for that id we assign 
      //the token to it.
      //now for the input, we need the commas, the full stops and the apostrophes
      //every other punctuation marks too. so when we extract a word from the 
      //input, we split it at these punctuations, special characters and then
      //separately do the mapping in them
      std::ifstream file(filePath);

      std::string token="";
      std::unordered_set<std::string> set;
      while(std::getline(file,token))
      {
           //now we split this individual token into sub-tokens consisting
           //of special characters, punctuations, numbers, any other things
           //std::cout<<token<<"\n";
           set.insert_range(Tokenizer::tokenize(token));
      }

      //now that we have the uniqueTokens from our data set, we can simply build the 
      //vocabulary, mapping the tokens to the unique ids
      uint32_t tokenId=0;
      for(const auto tokens:set)
      {
           vocabulary[tokens]=tokenId;
           decoderMap[tokenId]=tokens;
           tokenId++;
      }
      
      const uint32_t unkId=tokenId;
      tokenId++;

      vocabulary[UNK_TOKEN]=unkId;
      decoderMap[unkId]=UNK_TOKEN;

      std::cout<<"Printing vocabulary\n ";
      for(auto vals:vocabulary)
      {
           std::cout<<"Key = "<<vals.first<<" Val= "<<vals.second<<"\n";
      }
}

void Tokenizer::init(const std::string& filePath)
{
      buildVocabulary(filePath);
}

void Tokenizer::toPrint(std::set<std::string>& uniqueTokens)
{
      std::cout<<"Printing Token Ids= \n [";
      for(const std::string& tokens:uniqueTokens)
      {
           std::cout<<vocabulary[tokens]<<" ";
      }
      std::cout<<"] \n";
}

std::vector<uint32_t> Tokenizer::encoder(const std::vector<std::string>& data)
{
      std::vector<uint32_t> tokenIds;

      for(const std::string &text : data)
      {
           //tokenize the raw text sequentially
           std::vector<std::string> sequence = Tokenizer::tokenize(text);
           
           //map sequential strings directly to IDs
           for(const std::string& token : sequence)
           {
                //
                if(vocabulary.find(token)==vocabulary.end())
                {
                     tokenIds.push_back(vocabulary[UNK_TOKEN]);
                     continue;
                }
                tokenIds.push_back(vocabulary[token]);

           }
      } 
      return tokenIds;
}

//decoder method, to use the vocabulary and output the set of words
//along with the token ids
std::vector<std::string> Tokenizer::decoder(const std::vector<uint32_t>&tokenIds,
      bool toPrint)
{
      //for decoding the tokens, we can simply use the unque tokens
      //vector 
      std::vector<std::string> decodedTokens;
      for(const uint32_t &ids:tokenIds)
      {
           decodedTokens.emplace_back(decoderMap[ids]);
      }     
      return decodedTokens;
}
