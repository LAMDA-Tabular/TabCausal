# main script for training

######### training params
CUDA=0
NUM_GPU=1

######### data params

# NOTE: name of YAML file and run save folder
# see ./config for more options
TAG="aggregator_tf_fci"
#TAG="aggregator_tf_gies"
CONFIG="config/${TAG}.yaml"

# NOTE: customize this to your save folder.
# The trainer will make a timestamped subfolder within this directory.
SAVE_ROOT="${SEA_SAVE_ROOT:-runs/sc-baselines}"
SAVE_PATH="${SAVE_ROOT}/${TAG}"
# if you messed up and your job died, uncomment this and --checkpoint_path
#CKPT_PATH=""

python src/train.py \
    --config_file $CONFIG \
    --save_path $SAVE_PATH \
    --gpu $CUDA \
    --num_gpu $NUM_GPU \
    #--checkpoint_path $CKPT_PATH \
