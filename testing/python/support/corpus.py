"""Deterministic multilingual inputs shared by tests and benchmarks."""

import random

TEMPLATES = {
    "English": "The weather changed at {n}; tomorrow we'll go to Chikkamagaluru. It's a long drive.\n",
    "Bengali": "বাংলা ভাষায় লেখা পরীক্ষা। কলকাতা থেকে ঢাকায় গেলাম। উদাহরণ {n}।\n",
    "Hindi": "हिन्दी में परीक्षण। नई दिल्ली से बेंगलुरु तक यात्रा {n} किलोमीटर।\n",
    "Chinese": "快速测试分词器对中文文本的处理。样本 {n} 包含代码与数字。\n",
    "C++": "template<class T>\nT clamp(T x) {{ return x < {n} ? {n} : x; }}\r\n",
    "Java": "public record Sample(int id) {{ int scale() {{ return id * {n}; }} }}\n",
    "Python": "def sample(values):\n\treturn [x**2 for x in values if x > {n}]\n",
    "Math/JSON": '{{"n":{n},"math":"∑ᵢ xᵢ² ≤ √2; ١٢٣ ± π/2 → ∞","hex":"0xAF"}}\n',
    "Emoji/mixed": "👩🏽‍💻🧪🇮🇳🙂 #{n}: café / cafe\u0301; বাংলা; हिन्दी; 中文。\n",
}

EDGES = [
    "",
    "\0",
    "Aé🙂",
    "a\u0085b\u00a0c\u2003d\u2028e\u2029f\u3000g",
    "\u2581a",
    "\t\r\n  x\n \t \r y  ",
    "a’b 'S 'T 'RE 'VE 'M 'LL 'D 'ſ 'K",
    "a\U000323b0b",
    "1²३४৫ⅫⅧ",
    "a<|endoftext|>😀<|endofprompt|>b",
    "\u0301\u0300e\u0301 क्ष বাংলা",
    "🧑‍🚀❤️‍🔥 — ✨",
    "a\x1fb",
    "日本語𰻞",
    "𐦀\ua7cb",
    "  !\r\n\t \n ",
    "\u2060x\ufeffy",
    "hello world",
    "abcd",
]

FAMILIES = [
    "aA 'RE\n\t123_*𐦀ſK",
    "বাংলা ভাষায় কলকাতা ১২৩",
    "हिन्दी में क्ष त्र १२३४",
    "中文汉字拼音繁體中文。",
    "👨‍👩‍👦👩🏽‍💻🙂🇮🇳️",
    "₀₁₃−√≤∑π\u0085\u00a0\u0301\u2028",
]


def fixture():
    rng = random.Random(642815)
    documents = [
        (name, template.format(n=rng.randrange(100_000_000)))
        for _ in range(600)
        for name, template in TEMPLATES.items()
    ]
    random_text = []
    for i in range(6000):
        pool = FAMILIES[i % len(FAMILIES)] if i % 4 else "".join(FAMILIES)
        chars = []
        for _ in range(rng.randint(1, 96)):
            if rng.randrange(8) == 0:
                cp = rng.randrange(0x110000 - 0x800)
                chars.append(chr(cp + (0x800 if cp >= 0xD800 else 0)))
            else:
                chars.append(rng.choice(pool))
        random_text.append(("Random Unicode", "".join(chars)))
    return documents, random_text + [("Edge cases", x) for x in EDGES]


def exact_size(seed, n):
    result = (
        (seed * (n // len(seed)) + seed[: n % len(seed)])
        .decode("utf-8", "ignore")
        .encode()
    )
    return result.ljust(n, b" ")


def long_spans():
    return [
        ("letter run 64 KiB", b"a" * 65_536),
        ("space run 64 KiB", b" " * 65_536),
        ("punctuation run 64 KiB", b"!" * 65_536),
        ("emoji run 64 KiB", "🙂".encode() * 16_384),
    ]
