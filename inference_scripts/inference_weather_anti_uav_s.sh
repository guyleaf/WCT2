#!/usr/bin/env bash
set -e

root=$1
device="${2:-0}"

if [ -z "$root" ]
then
    echo "Please enter the root folder of dataset or -h, --help for help."
    exit 1
fi

if [ "$root" = "-h" ] || [ "$root" = "--help" ]
then
    echo "Usage: $0 ROOT_FOLDER [DEVICE, e.g. 4]"
    exit 0
fi

echo "Root folder: $root"
echo "Device: $device"

datasets=("FWID" "Image2Weather")
weathers=("foggy" "rainy" "snowy")

imageSize=1280
unpoolMethod="cat5"
outputPrefix="$root/stylized_wct2_${imageSize}_${unpoolMethod}_pad"

for dataset in "${datasets[@]}"
do
    content="$root/harmonized/${dataset}"
    style="$root/backgrounds/${dataset}"
    output="$outputPrefix/${dataset}"

    for weather in "${weathers[@]}"
    do
        CUDA_VISIBLE_DEVICES=$device python transfer.py --option_unpool "$unpoolMethod" -e -d -s --image_size $imageSize --content "$content/$weather" --style "$style/$weather" --output "$output/$weather"
    done

    cp -r "$content/clear" "$content/cloudy" "$output"
done
