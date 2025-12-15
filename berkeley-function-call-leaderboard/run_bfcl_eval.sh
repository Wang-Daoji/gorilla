#!/bin/bash

LIB="memos-api"
MODEL="gpt-4o-mini-2024-07-18-FC"
WORKERS=1

# laod env variables
export MEMOS_URL=http://0.0.0.0:8001 # memos-api server url
export VERSION="default_version"
export TOPK=10
export FRAME=$LIB

RESULT_DIR="results/${LIB}_${VERSION}"
mkdir -p "$RESULT_DIR"

SCORE_DIR="$RESULT_DIR/score"
mkdir -p "$SCORE_DIR"

DATA_DIR="data/bfcl-v4"

echo "================================"
echo "BFCL Evaluation Configuration:"
echo "================================"
echo "LIB: $LIB"
echo "MODEL: $MODEL"
echo "WORKERS: $WORKERS"
echo "VERSION: $VERSION"
echo "TOPK: $TOPK"
echo "FRAME: $FRAME"
echo "RESULT_DIR: $RESULT_DIR"
echo "SCORE_DIR: $SCORE_DIR"
echo "================================"
echo ""



echo "Running bfcl_ingestion.py..."
python bfcl_ingestion.py --lib $LIB --workers $WORKERS --data-dir $DATA_DIR --record-dir $RESULT_DIR
if [ $? -ne 0 ]; then
    echo "Error running bfcl_ingestion.py"
    exit 1
fi

echo "Running bfcl_generate.py..."
python openfunctions_evaluation.py --num-threads $WORKERS --model $MODEL --result-dir $RESULT_DIR --test-category multi_turn single_turn --run-ids
if [ $? -ne 0 ]; then
    echo "Error running bfcl_generate.py"
    exit 1
fi

echo "Running bfcl_evaluate.py..."
python bfcl_eval/eval_checker/eval_runner.py --model $MODEL --result-dir $RESULT_DIR --score-dir $SCORE_DIR --test-category  multi_turn single_turn --partial-eval
if [ $? -ne 0 ]; then
    echo "Error running bfcl_evaluate.py"
    exit 1
fi
