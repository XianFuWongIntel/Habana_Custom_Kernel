#!/bin/bash

cd ../
mkdir -p build && cd build

TEST_NUM=$1
IS_WEIGHT=$2

TEST_NUM=${TEST_NUM:=1}
IS_WEIGHT=${IS_WEIGHT:=0}
re='^[0-9]+$'

if [ $TEST_NUM == "all" ]; then
    TOTAL_TEST=$(ls ../test_binaries/*.json | wc -l)
    for i in $(seq $TOTAL_TEST); do
        if [ $IS_WEIGHT == "all" ]; then
            for j in 0 1; do
                chronic cmake -DCMAKE_BUILD_TYPE=Release -DTEST_NUM=$i -DIS_WEIGHT=$j ..
                chronic make clean
                chronic make -j$nproc
                ./tests/tpc_kernel_tests -t QuantizeFwdF32
            done
        elif [[ $IS_WEIGHT =~ $re ]]; then
            chronic cmake -DCMAKE_BUILD_TYPE=Release -DTEST_NUM=$i -DIS_WEIGHT=$IS_WEIGHT ..
            chronic make clean
            chronic make -j$nproc
            ./tests/tpc_kernel_tests -t QuantizeFwdF32
        fi
    done
elif [[ $TEST_NUM =~ $re ]]; then
    if [ $IS_WEIGHT == "all" ]; then
        for i in 0 1; do
            chronic cmake -DCMAKE_BUILD_TYPE=Release -DTEST_NUM=$TEST_NUM -DIS_WEIGHT=$i ..
            chronic make clean
            chronic make -j$nproc
            ./tests/tpc_kernel_tests -t QuantizeFwdF32
        done
    elif [[ $IS_WEIGHT =~ $re ]]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DTEST_NUM=$TEST_NUM -DIS_WEIGHT=$IS_WEIGHT ..
        make clean && make -j$nproc
        ./tests/tpc_kernel_tests -t QuantizeFwdF32
    fi
fi
