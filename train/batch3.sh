cd "$(dirname "$0")"
MODEL=nova-1.9 EPOCHS=30 MAX_SEQS=3600 node train.js > tf-b3-19.log 2>&1
MODEL=supernova-1.0 EPOCHS=30 MAX_SEQS=3600 node train.js > tf-b3-sn.log 2>&1
echo "BATCH3-DONE $(date)" >> tf-batch.log
