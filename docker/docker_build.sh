sudo docker run --gpus all -it \
	-u 0 \
	-p 1226:7479 \
	--ipc=host \
	--name cuml \
	-v /home/doyoon:/home/doyoon \
	doyooni303/cuml
