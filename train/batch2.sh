cd "$(dirname "$0")"
MODEL=nova-1.5 EPOCHS=26 MAX_SEQS=3200 node train.js > tf-b2-15.log 2>&1
MODEL=supernova-1.0 EPOCHS=26 MAX_SEQS=3200 node train.js > tf-b2-sn.log 2>&1
echo "BATCH2-DONE $(date)" >> tf-batch.log
