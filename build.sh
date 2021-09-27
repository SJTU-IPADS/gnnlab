#!/bin/bash
## ./build.sh [-c]
##      -c for clean build
export MAX_JOBS=40
if [ "$1" == '-c' ]; then
echo "Cleaning..."
python3 setup.py clean -q | \
  grep -v '^g\+\+' | \
  grep -v '^[a-zA-Z\-]*ed ' | \
  grep -v '^[a-zA-Z\-]*ing '
fi
echo "Building..."
python3 setup.py build  -q | \
  grep -v -e '^\[[0-9]*/[0-9]*\]' | \
  grep -v 'valid for C/ObjC but not for C++' | \
  grep -v '^g\+\+' | \
  grep -v '^[a-zA-Z\-]*ed ' | \
  grep -v '^[a-zA-Z\-]*ing '
echo "Installing..."
python3 setup.py install -q | \
  grep -v -e '^\[[0-9]*/[0-9]*\]' | \
  grep -v '^g\+\+' | \
  grep -v '^[a-zA-Z\-]*ed ' | \
  grep -v '^[a-zA-Z\-]*ing '
echo "Done."
