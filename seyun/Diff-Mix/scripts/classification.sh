#dataset    (dog         cub        car         aircraft      flower      pet      food     chest     caltech   pascal  )
#lr(resnet) (0.001       0.05       0.10        0.10          0.05        0.01     0.01     0.01      0.01      0.01    )
#lr(vit)    (0.00001     0.001      0.001       0.001         0.0005      0.0001   0.00005  0.00005   0.00005   0.00005 )

GPU=1
DATASET="cub"
SHOT=-1
# "shot{args.examples_per_class}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"
SYNDATA_DIR="aug_samples/cub/shot${SHOT}_diff-mix_fixed_0.7" # shot-1 denotes full shot
SYNDATA_P=0.1
GAMMA=0.8

python downstream_tasks/train_hub.py \
    --dataset $DATASET \
    --syndata_dir $SYNDATA_DIR \
    --syndata_p $SYNDATA_P \
    --model "resnet50" \
    --gamma $GAMMA \
    --examples_per_class $SHOT \
    --gpu $GPU \
    --amp 2 \
    --note $(date +%m%d%H%M) \
    --group_note "fullshot" \
    --nepoch 120 \
    --res_mode 224 \
    --lr 0.05 \
    --seed 0 \
    --weight_decay 0.0005 