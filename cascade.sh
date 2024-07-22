sh script/train/finetune_full.sh > siglip0.4B-yi1.5-9B-Chat-visionPretrained-alignment-sft2-V13.log 2>&1
sh script/train/pretrain.sh > 34B-pretrain.log 2>&1
sh script/train/finetune_lora.sh > siglip0.4B-yi1.5-34B-Chat-visionPretrained-alignment-sft2-V13.log 2>&1
sh zk.sh


