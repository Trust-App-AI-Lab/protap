results_path="./test"  # replace to your results path
batch_size=8
weight_path="../DrugCLIP/checkpoint_best.pt"

TASK="PCBA" # DUDE or PCBA

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python test_DrugCLIP.py --user-dir ./prot_learn/models/drugclip/unimol $data_path '../DrugCLIP/data/train_no_test_af' --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \


