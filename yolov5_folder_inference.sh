#!/usr/bin/env bash


# usage : ./yolov5_inference.sh [video folder_path] [weights_name] [--view_img] [--Save_annotations]
# weights_name: anything from (yolov5s.pt, yolov5l.pt, tl_yolov5s_best.pt, tl_yolov5l_best.pt) 

cd $1
for F in */;
do
echo "${F}";
if [ -d "$1/$F/inference_result" ]
then
    echo "Directory $1/$F/inference_result exists.";
else
    mkdir $1/$F/inference_result;
fi

cd ../Pedestrian_trajectory_prediction
python3 inference.py --input $1/${F}/imgs_without_time_sync --model_weights $2 --output $1/$F/inference_result/${F} $3 $4

cd $1

done
