#!/bin/bash

files=`ls _notebooks/*.ipynb`
for file in $files
do
  jupyter nbconvert --to markdown $file
  file_name=`basename $file .ipynb`
  rm "_posts/${file_name}.md"
  rm -r "assets/images/${file_name}_files"
  mv _notebooks/"$file_name".md _posts
  mv _notebooks/"$file_name"_files assets/images/
  sed -i "s|${file_name}_files|assets/images/${file_name}_files|g" _posts/"$file_name".md
done
