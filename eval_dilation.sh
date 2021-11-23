## declare an array variable
dilation_type=("INPUT" "OUTPUT")
interp_type=("cubic")
interp_ratio=(2 4 8 16 32)


#cpt_path=pretrained_models/humanact12/checkpoint_5000.pth.tar
base_config=configs/config_eval_dil.yaml
tmp_config=configs/config_tmp.yaml

# no interpolation
#python eval.py --cfg $base_config

## now loop through the above array
for inp in "${dilation_type[@]}"
do
  for t in "${interp_type[@]}"
  do
    for r in "${interp_ratio[@]}"
    do
      model_name="input_dil"
      if [ "$inp" = "OUTPUT" ]; then
          model_name="out_dil"
      fi
      model_dir="./results/${model_name}_${r}/"
      possible_models=( $(ls -d ${model_dir}*/) )
      num_models=${#possible_models[@]}
      if (( num_models > 1 )); then
        echo "More than one model in [${model_dir}] aborting!"
        exit 1
      fi
      model_dir="${possible_models[0]}"
      pretrained_path="${model_dir}model_best.pth.tar"
      echo "Running with model [${pretrained_path}]"

      cp $base_config $tmp_config
      sed -i "20i \ \ PRETRAINED: '${pretrained_path}'" $tmp_config
      echo "" >> $tmp_config
      echo "  ${inp}_DILATION_RATE: ${r}" >> $tmp_config
      echo "EVAL:" >> $tmp_config
      echo "  INTERP_TYPE: '${t}'" >> $tmp_config
      python eval.py --cfg $tmp_config
    done
  done
done