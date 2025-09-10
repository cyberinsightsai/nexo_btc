#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title nexo-btc
# @raycast.mode compact

# Optional parameters:
# @raycast.icon 🤖

# Documentation:
# @raycast.description Best time to exchange
# @raycast.author andres_eduardo_alonso_guio
# @raycast.authorURL https://raycast.com/andres_eduardo_alonso_guio

cd /Users/andres/code/cyberinsights/nexo_btc
source /System/Volumes/Data/Users/andres/code/streamlit-map-test/.venv/bin/activate
python3 nexo_btc_exchange.py > /tmp/python_output.txt
output=$(tail -1 /tmp/python_output.txt)
osascript <<EOF
set output to "$output"
display notification  "$output" with title "Output" subtitle "Result from
 Python Script" sound name "Glass"
EOF
