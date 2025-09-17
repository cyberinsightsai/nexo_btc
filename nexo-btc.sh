#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Nexo-BTC Ratio
# @raycast.mode compact

# Optional parameters:
# @raycast.icon 🤖

# Documentation:
# @raycast.description Explore if its a good oportunity to swap nexo-btc
# @raycast.author Andrés

cd /Users/andres/code/crypto/nexo-btc
source /Users/andres/code/crypto/nexo-btc/.venv/bin/activate
python3 nexo_btc_exchange.py > /tmp/python_output.txt
output=$(tail -1 /tmp/python_output.txt)
osascript <<EOF
set output to "$output"
display notification  "$output" with title "Output" subtitle "Result from
 Python Script" sound name "Glass"
EOF
osascript -e 'tell app "System Events" to display dialog "'$output'"'
