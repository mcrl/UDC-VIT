#!/bin/bash

# MIN_DIR=0 MAX_DIR=1000 REMOTE_DIR=/home/admin/udc-vit/result/ ./scp_raspi2ds.sh
# MIN_DIR=0 MAX_DIR=1000 REMOTE_DIR='your host' SSH_PORT=22 ./scp_raspi2ds.sh


SSH_USER=admin
SSH_HOST='your sshhost' 
SSH_PORT=12800
REMOTE_DIR=/home/admin/udc-vit/result/
LOCAL_DIR=UDC-VIT-Work/alignment/yuv/

set -a
: ${MIN_DIR=0}
: ${MAX_DIR=0}
: ${SSH_USER}=admin
: ${SSH_HOST}="your host"
: ${SSH_PORT}=12800
: ${REMOTE_DIR}=/home/admin/udc-vit/result/
: ${LOCAL_DIR}=UDC-VIT-Work/alignment/yuv/
set +a

if [ -z "$min" ]; then
    min=0
fi

if [ -z "$max" ]; then
    max=1000  
fi


folders=$(ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "find $REMOTE_DIR -mindepth 1 -maxdepth 1 -type d -exec basename {} \;" | sort -n)

for folder in $folders; do
    folder_number=$folder
    
    if [ -n "$folder_number" ]; then

        if [ "$folder_number" -eq "$folder_number" ] 2>/dev/null; then
            if [ "$folder_number" -ge "$min" ] && [ "$folder_number" -le "$max" ]; then
                scp -r -P $SSH_PORT $SSH_USER@$SSH_HOST:$REMOTE_DIR$folder $LOCAL_DIR
            fi
        else
            echo "Folder number is empty: $folder"
        fi
    else
        echo "Folder number is not a number: $folder"
    fi
done
