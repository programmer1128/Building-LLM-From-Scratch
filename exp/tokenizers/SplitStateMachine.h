#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

class SplitStateMachine
{
    private:
    enum class CharCategory : uint8_t
    {
        NEWLINE,    // '\r' or '\n'
        WHITESPACE, // '\s' excluding '\r' and '\n'
        LETTER,     // \p{L}
        NUMBER,     // \p{N}
        SYMBOL      // Everything else: [^\s\p{L}\p{N}]
    };

    struct CodePoint
    {
        char32_t cp;
        uint8_t len;
    };

    struct CharInfo
    {
        CharCategory cat;
        uint8_t len;
    };

    struct CodePointRange
    {
        char32_t start;
        char32_t end;
    };

    static CharCategory asciiCategory(unsigned char b);
    static CodePoint nextCodepoint(std::string_view data, size_t pos);
    static bool isUnicodeLetter(char32_t cp);
    static bool isUnicodeNumber(char32_t cp);
    static CharCategory classify(char32_t cp);
    static CharInfo classifyAt(std::string_view data, size_t pos);

    // Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    static size_t matchContraction(std::string_view data, size_t pos);

    // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
    static size_t matchWords(std::string_view data, size_t pos, CodePoint cp0, CharCategory cat0);

    // Rule 3: \p{N}{1,3}
    static size_t matchNumbers(std::string_view data, size_t pos, CodePoint cp0, CharCategory cat0);

    // Rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
    static size_t matchSymbols(std::string_view data, size_t pos, CodePoint cp0, CharCategory cat0);

    // Rules 5, 6, and 7: \s*[\r\n]+ | \s+(?!\S) | \s+
    static size_t matchWhitespace(std::string_view data, size_t pos, CodePoint cp0, CharCategory cat0);

     public:
    static std::vector<std::string_view> split(std::string_view data);
};
