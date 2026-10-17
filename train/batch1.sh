cd "$(dirname "$0")"
MODEL=nova-1.5 EPOCHS=26 MAX_SEQS=3200 node train.js > tf-batch-15.log 2>&1
MODEL=supernova-1.0 EPOCHS=26 MAX_SEQS=3200 node train.js > tf-batch-sn.log 2>&1
echo "BATCH1-DONE $(date)" >> tf-batch.log
