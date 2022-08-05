#실행 예시 sh docker_build.sh bb451/dacon
docker build -t $1 --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
