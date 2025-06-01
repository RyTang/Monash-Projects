#/bin/sh

## Make directory
mkdir data

## Download images
wget https://storage.googleapis.com/ads-dataset/subfolder-0.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-1.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-2.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-3.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-4.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-5.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-6.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-7.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-8.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-9.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-10.zip

## Download annotation file
wget https://people.cs.pitt.edu/~kovashka/ads/annotations_images.zip

## Unzip folders
unzip subfolder-0.zip -d data/
unzip subfolder-1.zip -d data/
unzip subfolder-2.zip -d data/
unzip subfolder-3.zip -d data/
unzip subfolder-4.zip -d data/
unzip subfolder-5.zip -d data/
unzip subfolder-6.zip -d data/
unzip subfolder-7.zip -d data/
unzip subfolder-8.zip -d data/
unzip subfolder-9.zip -d data/
unzip subfolder-10.zip -d data/
unzip annotations_images.zip -d data/

## Rename annotation folder
mv data/image data/annotations

## Remove all zip files
rm subfolder-0.zip
rm subfolder-1.zip
rm subfolder-2.zip
rm subfolder-3.zip
rm subfolder-4.zip
rm subfolder-5.zip
rm subfolder-6.zip
rm subfolder-7.zip
rm subfolder-8.zip
rm subfolder-9.zip
rm subfolder-10.zip
rm annotations_images.zip

exit 0