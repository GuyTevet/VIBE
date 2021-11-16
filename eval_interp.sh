## declare an array variable
interp_type=("nearest" "linear" "cubic")
interp_ratio=(2 3 4 5 6 8 10 12 14 16)

#cpt_path=pretrained_models/humanact12/checkpoint_5000.pth.tar
base_config=configs/config_eval_seq_len_64.yaml
tmp_config=configs/config_tmp.yaml

# no interpolation
python eval.py --cfg $base_config

## now loop through the above array
for t in "${interp_type[@]}"
do
  for r in "${interp_ratio[@]}"
  do
    cp $base_config $tmp_config
    echo "" >> $tmp_config
    echo "EVAL:" >> $tmp_config
    echo "  INTERP_TYPE: '${t}'" >> $tmp_config
    echo "  INTERP_RATIO: ${r}" >> $tmp_config
    python eval.py --cfg $tmp_config
  done
done