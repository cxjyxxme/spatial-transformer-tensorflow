CUDA_VISIBLE_DEVICES="$1" python -u deploy_flow_rev.py \
					--model-dir ./models/${3}/ --model-name model-${2} --before-ch 31 \
					--output-dir ./dataset/${3}-${2}/Parallax  --test-list /home/ubuntu/Parallax/Parallax/list.txt --indices 7 13 19 25 31 --prefix /home/ubuntu/Parallax/Parallax;
CUDA_VISIBLE_DEVICES="$1" python -u deploy_flow_rev.py \
					--model-dir ./models/${3}/ --model-name model-${2} --before-ch 31 \
 					--output-dir ./dataset/${3}-${2}/Crowd  --test-list /home/ubuntu/Crowd/Crowd/list.txt --indices 7 13 19 25 31 --prefix /home/ubuntu/Crowd/Crowd;
CUDA_VISIBLE_DEVICES="$1" python -u deploy_flow_rev.py \
					--model-dir ./models/${3}/ --model-name model-${2} --before-ch 31 \
					--output-dir ./dataset/${3}-${2}/Zooming  --test-list /home/ubuntu/Zooming/Zooming/list.txt --indices 7 13 19 25 31 --prefix /home/ubuntu/Zooming/Zooming
