

#ROOT = "../"

SAVE_PATH="./data-generation/datasets/generated-20ep/"
TASKS=("cube_container" "object_container" "container_place")
IMG_SIZE="224 224"
PROCESSES=1
EP_TASK=20
VARIATIONS=-1
ARM_ACTION_MODE="JointVelocity" # Even if we add custom environment arm action mode in dataset_generator, the observations still have gripper tip position wrt world


python RLBench/rlbench/dataset_generator.py \
    --save_path $SAVE_PATH \
    --tasks ${TASKS[@]} \
    --image_size $IMG_SIZE \
    --processes $PROCESSES \
    --episodes_per_task $EP_TASK \
    --variations $VARIATIONS \
    --arm_action_mode $ARM_ACTION_MODE \
    #--renderer $VAR \
    #--arm_max_velocity $VAR \
    #--arm_max_acceleration $VAR 
