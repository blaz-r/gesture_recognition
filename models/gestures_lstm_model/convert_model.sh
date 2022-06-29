# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh
# source /opt/intel/openvino_2021/bin/setupvars.sh

convert_model () {
	model_name=$1
#	tflite2tensorflow \
#		--model_path ${model_name}.tflite \
#		--model_output_path ${model_name} \
#		--flatc_path ../../../flatc \
#		--schema_path ../../../schema.fbs \
#		--output_pb \
#    	--optimizing_for_openvino_and_myriad \
#        --rigorous_optimization_for_myriad

	# For generating Openvino "non normalized input" models (the normalization would need to be made explictly in the code):
	#tflite2tensorflow \
	#  --model_path ${model_name}.tflite \
	#  --model_output_path ${model_name} \
	#  --flatc_path ../../flatc \
	#  --schema_path ../../schema.fbs \
	#  --output_openvino_and_myriad

#	 Generate Openvino "normalized input" models
#	/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py \
#		--saved_model_dir ${model_name} \
#		--data_type FP16 \
#		--model_name ${model_name} \
#		--input input \
#		--input_shape [1,30,42]
#
  /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py \
  --input_model ${model_name}.pb \
  --data_type FP16 \
#  --input "sequential/lstm/TensorArrayUnstack/TensorListFromTensor" \
#  --input_shape [30,42]

#  python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_onnx.py \
#                --input_model ${model_name}.onnx --data_type half \
#                --log_level=DEBUG

	# For Interpolate layers, replace in coordinate_transformation_mode, "half_pixel" by "align_corners"  (bug optimizer)
	# replace ${model_name}.xml half_pixel align_corners

	/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile \
		-m ${model_name}.xml \
		-ip FP16 \
		-VPU_NUMBER_OF_SHAVES 4 \
		-VPU_NUMBER_OF_CMX_SLICES 4 \
		-o ${model_name}_sh4.blob
}

#convert_model gesture_recognition_lstm
convert_model gesture_recognition_lstm_tf