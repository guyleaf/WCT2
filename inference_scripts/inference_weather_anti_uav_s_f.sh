#!/usr/bin/env bash
set -e

content_root=$1
style_root=$2
device="${3:-0}"

if [ -z "$content_root" ]
then
    echo "Please enter the content root folder of dataset or -h, --help for help."
    exit 1
fi
if [ -z "$style_root" ]
then
    echo "Please enter the style root folder of dataset or -h, --help for help."
    exit 1
fi

if [ "$content_root" = "-h" ] || [ "$content_root" = "--help" ]
then
    echo "Usage: $0 CONTENT_FOLDER STYLE_FOLDER [DEVICE, e.g. 4]"
    exit 0
fi

echo "Content root folder: $content_root"
echo "Style root folder: $style_root"
echo "Device: $device"

datasets=("FWID" "Image2Weather")
weathers=("foggy" "rainy" "snowy")

imageSize=1280
unpoolMethod="cat5"
outputPrefix="$content_root/stylized_wct2_${imageSize}_${unpoolMethod}_pad"

for dataset in "${datasets[@]}"
do
    content="$content_root/harmonized/${dataset}_fake_style"
    style=$style_root
    output="$outputPrefix/${dataset}_fake_style"

    for weather in "${weathers[@]}"
    do
        CUDA_VISIBLE_DEVICES=$device python transfer.py --option_unpool "$unpoolMethod" --option_mode "random" -e -d -s --image_size $imageSize --content "$content/$weather" --style "$style/$weather" --output "$output/$weather"
    done

    content="$content_root/harmonized/${dataset}"
    output="$outputPrefix/${dataset}"
    mkdir -p "$output"
    cp -r "$content/clear" "$content/cloudy" "$output"
done
