#!/usr/bin/env bash

command_to_execute=$1
top
if [ -z "$command_to_execute" ]
then
    cd /opt/app
    if [[ "$AUTO_RELOAD" = true ]]
    then
        exec uvicorn kitmicroservices.framework:app --host 0.0.0.0 --reload
    else
        exec uvicorn kitmicroservices.framework:app --host 0.0.0.0
    fi
elif [ "$command_to_execute" = "start_worker" ]
then
    if [ "$AUTO_RELOAD" = true ] && [ -z "$MP_REMOTE_DEBUGGER" ]
    then
        # Note: microservice.yaml is mounted as file and it seems not working well with watchmedo
        # However, this yml/yaml will work for files in mounted directory
        exec watchmedo auto-restart -d . -p "*.py;*.yaml;*.yml" -R --signal SIGTERM python -- -m kitmicroservices.framework.start_worker
    else
        exec python -m kitmicroservices.framework.start_worker
    fi
else
    exec "$@"
fi