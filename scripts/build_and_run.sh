#!/bin/bash

CONFIG_NUM=$1
IS_WEIGHT=$2
LAYER_NUM=$3
MODE=$4
BUILD_DIR=$5

CONFIG_NUM=${CONFIG_NUM:=1}
LAYER_NUM=${LAYER_NUM:=1}

IS_WEIGHT=${IS_WEIGHT:=0}
MODE=${MODE:="QuantizeFwdF32"}
BUILD_DIR=${BUILD_DIR:="build"}
LOG_PREFIX=${LOG_PREFIX:=""}

re='^[0-9]+$'

CONFIG_LIST=( 
    "qio_rn18-4bit-wq-sym-aq-asym-per-channel"
    "qio_rn18-8bit-wq-asym-aq-asym-per-tensor"
    "qio_rn18-8bit-wq-asym-aq-sym-per-channel"
    "qio_rn18-8bit-wq-sym-aq-asym-per-channel"
    "qio_rn18-8bit-wq-sym-aq-sym-per-tensor"
)

WT_LAYER_LIST=(
    "ResNet/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer1]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer1]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT"
    "ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT"
    "ResNet/NNCFLinear[fc]/linear_0|WEIGHT"
)

ACT_LAYER_LIST=(
    "ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer1]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer1]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/NNCFBatchNorm2d[1]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/NNCFBatchNorm2d[1]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/NNCFBatchNorm2d[1]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/AdaptiveAvgPool2d[avgpool]/adaptive_avg_pool2d_0|OUTPUT"
    "ResNet/ReLU[relu]/relu__0|OUTPUT"
    "ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT"
    "ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT"
)

TOTAL_LAYER_LIST=(
    ${#ACT_LAYER_LIST[@]}
    ${#WT_LAYER_LIST[@]}
)

cd ../
mkdir -p $BUILD_DIR && cd $BUILD_DIR
if [ $CONFIG_NUM == "all" ]; then
    for i in "${!CONFIG_LIST[@]}"; do 
        echo "===================================================================="
        echo "Config $((i+1)) : ${CONFIG_LIST[$i]}" - $MODE
        echo "===================================================================="
        if [ $IS_WEIGHT == "all" ]; then
            for j in 1 0; do
                if [ $j == 0 ]; then echo "Activation Test"; else echo "Weight Test"; fi
                if [ $LAYER_NUM == "all" ]; then
                    cd .. && cd scripts
                    mkdir -p ../test_log
                    opt_nproc=$(($(nproc)/(${#CONFIG_LIST[@]} * ${#TOTAL_LAYER_LIST[@]})))
                    (time nproc=$((opt_nproc < 1 ? 1 : opt_nproc)) ./build_and_run.sh $((i+1)) $j all $MODE build_$((i+1))_$j) > ../test_log/$LOG_PREFIX$((i+1))_$j.log 2>&1 &
                    echo "PID: $!"
                    echo "Log: $(realpath ../test_log/$LOG_PREFIX$((i+1))_$j.log)"
                elif [[ $LAYER_NUM =~ $re ]]; then
                    if [ $j == 0 ]; then echo -n "${ACT_LAYER_LIST[$((LAYER_NUM-1))]} : "; else echo -n "${WT_LAYER_LIST[$((LAYER_NUM-1))]} : "; fi
                    chronic cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$((i+1)) -DLAYER_NUM=$LAYER_NUM -DIS_WEIGHT=$j ..
                    chronic make clean
                    chronic make -j$nproc
                    SECONDS=0
                    OUTPUT_LOG=$(./tests/tpc_kernel_tests -t $MODE)
                    RESULT=$(echo "$OUTPUT_LOG" | grep Quantize)
                    ELEM=$(echo "$OUTPUT_LOG" | grep "Element" | cut -d ":" -f2)
                    MAX_ABS=$(echo "$OUTPUT_LOG" | grep "Max Abs" | cut -d ":" -f2)
                    MAX_REL=$(echo "$OUTPUT_LOG" | grep "Max Rel" | cut -d ":" -f2)
                    echo "$RESULT Elapsed(s): $SECONDS Element: $ELEM Max Abs: $MAX_ABS Max Rel: $MAX_REL"
                fi
                echo ""
            done
        elif [[ $IS_WEIGHT =~ $re ]]; then
            if [ $IS_WEIGHT == 0 ]; then echo "Activation Test"; else echo "Weight Test"; fi
            if [ $LAYER_NUM == "all" ]; then
                echo "Running ${TOTAL_LAYER_LIST[$IS_WEIGHT]} test(s) ..."
                for j in $(seq ${TOTAL_LAYER_LIST[$IS_WEIGHT]}); do
                    echo -n "Layer $j : "
                    chronic cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$((i+1)) -DLAYER_NUM=$j -DIS_WEIGHT=$IS_WEIGHT ..
                    chronic make clean
                    chronic make -j$nproc
                    SECONDS=0
                    OUTPUT_LOG=$(./tests/tpc_kernel_tests -t $MODE)
                    RESULT=$(echo "$OUTPUT_LOG" | grep Quantize)
                    ELEM=$(echo "$OUTPUT_LOG" | grep "Element" | cut -d ":" -f2)
                    MAX_ABS=$(echo "$OUTPUT_LOG" | grep "Max Abs" | cut -d ":" -f2)
                    MAX_REL=$(echo "$OUTPUT_LOG" | grep "Max Rel" | cut -d ":" -f2)
                    echo "$RESULT Elapsed(s): $SECONDS Element: $ELEM Max Abs: $MAX_ABS Max Rel: $MAX_REL"
                done
            elif [[ $LAYER_NUM =~ $re ]]; then
                if [ $IS_WEIGHT == 0 ]; then echo -n "${ACT_LAYER_LIST[$((LAYER_NUM-1))]} : "; else echo -n "${WT_LAYER_LIST[$((LAYER_NUM-1))]} : "; fi
                chronic cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$((i+1)) -DLAYER_NUM=$LAYER_NUM -DIS_WEIGHT=$IS_WEIGHT ..
                chronic make clean
                chronic make -j$nproc
                SECONDS=0
                OUTPUT_LOG=$(./tests/tpc_kernel_tests -t $MODE)
                RESULT=$(echo "$OUTPUT_LOG" | grep Quantize)
                ELEM=$(echo "$OUTPUT_LOG" | grep "Element" | cut -d ":" -f2)
                MAX_ABS=$(echo "$OUTPUT_LOG" | grep "Max Abs" | cut -d ":" -f2)
                MAX_REL=$(echo "$OUTPUT_LOG" | grep "Max Rel" | cut -d ":" -f2)
                echo "$RESULT Elapsed(s): $SECONDS Element: $ELEM Max Abs: $MAX_ABS Max Rel: $MAX_REL"
            fi
            echo ""
        fi
    done
elif [[ $CONFIG_NUM =~ $re ]]; then
    echo "===================================================================="
    echo "Config $CONFIG_NUM : ${CONFIG_LIST[$((CONFIG_NUM-1))]} - $MODE"
    echo "===================================================================="
    if [ $IS_WEIGHT == "all" ]; then
        for i in 1 0; do
            if [ $i == 0 ]; then echo "Activation Test"; else echo "Weight Test"; fi
            if [ $LAYER_NUM == "all" ]; then
                echo "Running ${TOTAL_LAYER_LIST[$i]} test(s) ..."
                for j in $(seq ${TOTAL_LAYER_LIST[$i]}); do
                    echo -n "Layer $j : "
                    chronic cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$CONFIG_NUM -DLAYER_NUM=$j -DIS_WEIGHT=$i ..
                    chronic make clean
                    chronic make -j$nproc
                    SECONDS=0
                    OUTPUT_LOG=$(./tests/tpc_kernel_tests -t $MODE)
                    RESULT=$(echo "$OUTPUT_LOG" | grep Quantize)
                    ELEM=$(echo "$OUTPUT_LOG" | grep "Element" | cut -d ":" -f2)
                    MAX_ABS=$(echo "$OUTPUT_LOG" | grep "Max Abs" | cut -d ":" -f2)
                    MAX_REL=$(echo "$OUTPUT_LOG" | grep "Max Rel" | cut -d ":" -f2)
                    echo "$RESULT Elapsed(s): $SECONDS Element: $ELEM Max Abs: $MAX_ABS Max Rel: $MAX_REL"
                done
            elif [[ $LAYER_NUM =~ $re ]]; then
                if [ $i == 0 ]; then echo -n "${ACT_LAYER_LIST[$((LAYER_NUM-1))]} : "; else echo -n "${WT_LAYER_LIST[$((LAYER_NUM-1))]} : "; fi
                chronic cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$CONFIG_NUM -DLAYER_NUM=$LAYER_NUM -DIS_WEIGHT=$i ..
                chronic make clean
                chronic make -j$nproc
                SECONDS=0
                OUTPUT_LOG=$(./tests/tpc_kernel_tests -t $MODE)
                RESULT=$(echo "$OUTPUT_LOG" | grep Quantize)
                ELEM=$(echo "$OUTPUT_LOG" | grep "Element" | cut -d ":" -f2)
                MAX_ABS=$(echo "$OUTPUT_LOG" | grep "Max Abs" | cut -d ":" -f2)
                MAX_REL=$(echo "$OUTPUT_LOG" | grep "Max Rel" | cut -d ":" -f2)
                echo "$RESULT Elapsed(s): $SECONDS Element: $ELEM Max Abs: $MAX_ABS Max Rel: $MAX_REL"
            fi
            echo ""
        done
    elif [[ $IS_WEIGHT =~ $re ]]; then
        if [ $IS_WEIGHT == 0 ]; then echo "Activation Test"; else echo "Weight Test"; fi
        if [ $LAYER_NUM == "all" ]; then
            echo "Running ${TOTAL_LAYER_LIST[$IS_WEIGHT]} test(s) ..."
            for i in $(seq ${TOTAL_LAYER_LIST[$IS_WEIGHT]}); do
                echo -n "Layer $i : "
                chronic cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$CONFIG_NUM -DLAYER_NUM=$i -DIS_WEIGHT=$IS_WEIGHT ..
                chronic make clean
                chronic make -j$nproc
                SECONDS=0
                OUTPUT_LOG=$(./tests/tpc_kernel_tests -t $MODE)
                RESULT=$(echo "$OUTPUT_LOG" | grep Quantize)
                ELEM=$(echo "$OUTPUT_LOG" | grep "Element" | cut -d ":" -f2)
                MAX_ABS=$(echo "$OUTPUT_LOG" | grep "Max Abs" | cut -d ":" -f2)
                MAX_REL=$(echo "$OUTPUT_LOG" | grep "Max Rel" | cut -d ":" -f2)
                echo "$RESULT Elapsed(s): $SECONDS Element: $ELEM Max Abs: $MAX_ABS Max Rel: $MAX_REL"
            done
        elif [[ $LAYER_NUM =~ $re ]]; then
            cmake -DCMAKE_PREFIX_PATH=../libtorch -DCMAKE_BUILD_TYPE=Release -DCONFIG_NUM=$CONFIG_NUM -DLAYER_NUM=$LAYER_NUM -DIS_WEIGHT=$IS_WEIGHT ..
            make clean && make -j$nproc
            ./tests/tpc_kernel_tests -t $MODE
        fi
    fi
fi
