# Building GPT from scratch in C++

This repository is a staged learning and implementation project: understand a
decoder-only transformer, implement it in C++, train and evaluate it, optimize
local inference, and connect it to a Java/Spring AI application.

Start with the [High-Level Design](model_building/docs/LLM-High-Level-Design-and-Innovation-Blueprint.md),
then review [LLD-01: Native Foundation Architecture](model_building/docs/LLD-01-Foundation-Architecture.md).
The [earlier roadmap](model_building/docs/ROADMAP.md) records the initial learning
plan; where its implementation choices differ, the HLD governs. The existing
`exp/tokenizers/` work is an experiment, not the native engine.
