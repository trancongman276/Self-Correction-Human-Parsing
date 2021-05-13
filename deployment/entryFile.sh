set -e

if [[ "$1" = "start_serv" ]]; then
    shift 1
    torchserve --start --ncs --ts-config /home/model-server/config.properties --model-store /home/model-server/model-store
else
    eval "$@"
fi

tail -f /dev/null