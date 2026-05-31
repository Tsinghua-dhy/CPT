#!/bin/bash

# Machine-1 launcher - processes split_1

echo "=========================================="
echo "Starting judgment for split 1"
echo "=========================================="

# Patch the config inside judge_mix_math_paths.py
python3 -c "
import re

# readfile
with open('judge_mix_math_paths.py', 'r', encoding='utf-8') as f:
    content = f.read()

# modifyinput_fileandoutput_file
content = re.sub(
    r'input_file = \".*?\"',
    'input_file = \"judged_pairs/pairs_data_split_1.jsonl\"',
    content
)
content = re.sub(
    r'output_file = \".*?\"',
    'output_file = \"judged_pairs/pairs_data_split_1_judged.jsonl\"',
    content
)

# Write the file back
with open('judge_mix_math_paths.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Configuration updated for split 1')
"

# Run the judge script
python3 judge_mix_math_paths.py

echo "=========================================="
echo "Split 1 judgment completed!"
echo "=========================================="
