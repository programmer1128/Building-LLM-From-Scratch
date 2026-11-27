cmake --build build --target dataloader_tests -j
ctest --test-dir build -R DataLoaderEdgeCasesTest --output-on-failure