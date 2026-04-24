#!/bin/bash
# iDragonfly Vast.ai Remote Manager

echo "Current balance:"
vastai show user --raw | jq -r '.credit'

action="$1"
vast_port=""
vast_host=""
instance_id=""

# Fetch metadata if variables are empty
if [ -z "$vast_host" ] || [ -z "$vast_port" ] || [ -z "$instance_id" ]; then
    # Parse the first active instance for host, port, and unique ID
    read vast_host vast_port instance_id <<< $(vastai show instances --raw | jq -r '.[0] | "\(.ssh_host) \(.ssh_port) \(.id)"')
    
    # Verify lookup was successful
    if [ -z "$vast_host" ] || [ "$vast_host" == "null" ]; then
        echo "Error: No active Vastai instance found or CLI not authenticated."
        exit 1
    fi
fi

echo "Instance ID: $instance_id"
echo "Target Host: $vast_host:$vast_port"
echo "ssh -p ${vast_port} -o ServerAliveInterval=60 root@${vast_host}" | tee ssh.log

if [ "$action" == "up" ]; then
    echo "Syncing local environment to remote..."
    ssh -p $vast_port root@$vast_host "mkdir -p /root/vast/"
    rsync -avz -e "ssh -p $vast_port -o ServerAliveInterval=60 -o ServerAliveCountMax=10" ./launch.sh setup.sh tune.sh fine_tune.sh eval.py eval2.py client.py requirements.txt trainer data data_tune data_fine tokenizer root@$vast_host:/root/vast/

elif [ "$action" == "destroy" ]; then
    echo "!!! WARNING: You are about to DESTROY instance $instance_id !!!"
    read -p "Are you sure you want to proceed? (y/N): " confirm
    if [[ "$confirm" == [yY] || "$confirm" == [yY][eE][sS] ]]; then
        echo "Destroying instance $instance_id..."
        vastai destroy instance $instance_id
    else
        echo "Aborted. Instance remains active."
    fi

elif [ "$action" == "logs" ]; then
    echo "Fetching logs..."
    rsync -avP -e "ssh -p $vast_port -o ServerAliveInterval=60 -o ServerAliveCountMax=10" root@$vast_host:/root/vast/*.log ./

elif [ "$action" == "down" ]; then
    echo "Syncing logs and metadata..."
    rsync -avP -e "ssh -p $vast_port -o ServerAliveInterval=60 -o ServerAliveCountMax=10" --include="*.log" --include="*.jsonl" --exclude="*" root@$vast_host:/root/vast/ ./
    echo "Syncing checkpoints (Resume enabled)..."
    rsync -avP -e "ssh -p $vast_port -o ServerAliveInterval=60 -o ServerAliveCountMax=10" root@$vast_host:/root/vast/checkpoints ./

else
    echo "Usage: $0 {up|down|logs|destroy}"
    echo "up:      Sync local code and data to remote"
    echo "down:    Sync remote checkpoints to local"
    echo "logs:    Sync remote .log files to local"
    echo "destroy: Terminate the Vast.ai instance immediately (with confirmation)"
fi
