CUDA_VISIBLE_DEVICES="$1" python -u deploy_flow_rev.py \
					--model-dir ./models/${3}/ --model-name model-${2} --before-ch 31 \
					--output-dir ./dataset/${3}-${2}/Running  --test-list /home/ubuntu/Running/Running/list.txt --indices 7 13 19 25 31 --prefix /home/ubuntu/Running/Running;
CUDA_VISIBLE_DEVICES="$1" python -u deploy_flow_rev.py \
					--model-dir ./models/${3}/ --model-name model-${2} --before-ch 31 \
 					--output-dir ./dataset/${3}-${2}/Regular  --test-list /home/ubuntu/Regular/Regular/list.txt --indices 7 13 19 25 31 --prefix /home/ubuntu/Regular/Regular;
CUDA_VISIBLE_DEVICES="$1" python -u deploy_flow_rev.py \
					--model-dir ./models/${3}/ --model-name model-${2} --before-ch 31 \
					--output-dir ./dataset/${3}-${2}/QuickRotation  --test-list /home/ubuntu/QuickRotation/QuickRotation/list.txt --indices 7 13 19 25 31 --prefix /home/ubuntu/QuickRotation/QuickRotation
