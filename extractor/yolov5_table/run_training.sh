#!/bin/bash

# Run the training script with nohup
nohup python yolov5/train.py --img 640 --batch 16 --epochs 50 --data datasets/imerit/row_col.yml --weights yolov5s.pt --project results/runs --device 0 > temp_train_output.log 2>&1 &

# Function to move the log file to the correct exp{i} folder
move_log_file() {
  while true; do
    # Find the latest created exp{i} folder
    EXP_FOLDER=$(ls -td results/runs/exp*/ 2>/dev/null | head -1)
    
    # If the exp{i} folder is found, move the log file
    if [ -n "$EXP_FOLDER" ]; then
      mv temp_train_output.log "$EXP_FOLDER/train_output.log"
      break
    fi
    # Wait for a second before checking again
    sleep 1
  done
}

# Run the move_log_file function in the background
move_log_file &

# Return terminal to the user
disown
