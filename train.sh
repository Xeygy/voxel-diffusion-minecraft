accelerate launch demo.py \
  --dataset_name="huggan/smithsonian_butterflies_subset" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-flowers-64" \
  --train_batch_size=16 \
  --num_epochs=50 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=fp16