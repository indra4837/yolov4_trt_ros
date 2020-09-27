echo "YOLOv3 or YOLOv4. Input 3 or 4"
read model_type

echo "What is the input shape? 208/416/608"
read input_shape

if [[ $model_type == 3 ]]
then
    if [[ $input_shape == 208 ]]
    then
        echo "Creating yolov3-208.cfg and yolov3-208.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=208/' | sed -e '8s/height=608/height=208/' > yolov3-208.cfg
        ln -sf yolov3.weights yolov3-208.weights
    fi
    if [[ $input_shape == 416 ]]
    then
        echo "Creating yolov3-416.cfg and yolov3-416.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=416/' | sed -e '8s/height=608/height=416/' > yolov3-416.cfg
        ln -sf yolov3.weights yolov3-416.weights
    fi
    if [[ $input_shape == 608 ]]
    then
        echo "Creating yolov3-608.cfg and yolov3-608.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' > yolov3-608.cfg
        ln -sf yolov3.weights yolov3-608.weights
    fi
else
    if [[ $input_shape == 208 ]]
    then
        echo "Creating yolov4-208.cfg and yolov4-208.weights"
        cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=208/' | sed -e '8s/height=608/height=208/' > yolov4-208.cfg
        ln -sf yolov4.weights yolov4-208.weights
    fi
    if [[ $input_shape == 416 ]]
    then
        echo "Creating yolov4-416.cfg and yolov4-416.weights"
        cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=416/' | sed -e '8s/height=608/height=416/' > yolov4-416.cfg
        ln -sf yolov4.weights yolov4-416.weights
    fi
    if [[ $input_shape == 608 ]]
    then
        echo "Creating yolov4-608.cfg and yolov4-608.weights"
        cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' > yolov4-608.cfg
        ln -sf yolov4.weights yolov4-608.weights
    fi
fi

echo "How many categories are there?"
read category_num
model_name = "yolov" + $model_type + "-" + $input_shape

# convert from yolo to onnx
python3 yolo_to_onnx.py -m $model_name -c $category_num

echo "Done converting to .onnx"
echo "..."
echo "Now converting to .trt"

# convert from onnx to trt
python3 onnx_to_trt.py -m $model_name -c $category_num --verbose

echo "Conversion from yolo to trt done!"
