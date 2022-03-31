#!/bin/bash

if [ `cat /etc/hostname` == "iZ0xi7b9pbfh70yjxzxh1mZ" ]; then
  sudo /opt/clean_page_cache/run.sh
else
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
fi

python3 runner.py