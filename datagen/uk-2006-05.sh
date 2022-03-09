#!/bin/bash
RAW_DATA_DIR='/graph-learning/data-raw'
UK_RAW_DATA_DIR="${RAW_DATA_DIR}/uk-2006-05"
OUTPUT_DATA_DIR='/graph-learning/samgraph/uk-2006-05'

download(){
  mkdir -p ${UK_RAW_DATA_DIR}
  if [ ! -e "${UK_RAW_DATA_DIR}/uk-2006-05.graph" ]; then
    pushd ${UK_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/uk-2006-05/uk-2006-05.graph
    wget http://data.law.di.unimi.it/webdata/uk-2006-05/uk-2006-05.properties
    popd
  elif [ ! -e "${UK_RAW_DATA_DIR}/uk-2006-05.properties" ]; then
    pushd ${UK_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/uk-2006-05/uk-2006-05.properties
    popd
  else
    echo "Binary file already downloaded."
  fi
}

generate_coo(){
  download
  if [ ! -e "${UK_RAW_DATA_DIR}/coo.bin" ]; then
    java -cp ../utility/webgraph/target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "${UK_RAW_DATA_DIR}/uk-2006-05"
    java -cp ../utility/webgraph/target/webgraph-0.1-SNAPSHOT.jar ipads.samgraph.webgraph.WebgraphDecoder "${UK_RAW_DATA_DIR}/uk-2006-05"
    mv ${UK_RAW_DATA_DIR}/uk-2006-05_coo.bin ${UK_RAW_DATA_DIR}/coo.bin
  else
    echo "COO already generated."
  fi
}

generate_coo
mkdir -p ${OUTPUT_DATA_DIR}
cat << EOF > ${OUTPUT_DATA_DIR}/meta.txt
NUM_NODE      77741046
NUM_EDGE      2965197340
FEAT_DIM      256
NUM_CLASS     150
NUM_TRAIN_SET 1000000
NUM_VALID_SET 200000
NUM_TEST_SET  100000
EOF

../utility/data-process/build/coo-to-dataset -g uk-2006-05
