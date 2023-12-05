#!/bin/bash

print_header() {
  local title="$1"
  local logfile="${2:-/dev/null}"  # Use /dev/null as the default logfile
  local line="================================================================================"
  local padding="                                                                               "
  local title_length=${#title}
  local padding_length=$(( 74 - title_length ))
  local left_padding_length=$(( padding_length / 2 ))
  local left_padding="${padding:0:left_padding_length}"
  local right_padding_length=$(( 74 - title_length - left_padding_length ))
  local right_padding="${padding:0:right_padding_length}"
  
  # Print to the console
  echo "$line"
  echo "===${left_padding}${title}${right_padding}==="
  echo "$line"

  # Append to the logfile if it's not /dev/null
  if [ "$logfile" != "/dev/null" ]; then
    echo "$line" >> "$logfile"
    echo "===${left_padding}${title}${right_padding}===" >> "$logfile"
    echo "$line" >> "$logfile"
  fi
}

