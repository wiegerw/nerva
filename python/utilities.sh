#!/bin/bash

print_header() {
  local title="$1"
  local line="================================================================================"
  local padding="                                                                               "
  local title_length=${#title}
  local padding_length=$(( 74 - title_length ))
  local left_padding_length=$(( padding_length / 2 ))
  local left_padding="${padding:0:left_padding_length}"
  local right_padding_length=$(( 74 - title_length - left_padding_length ))
  local right_padding="${padding:0:right_padding_length}"
  echo "$line"
  echo "===${left_padding}${title}${right_padding}==="
  echo "$line"
}
