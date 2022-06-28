# About this branch
This branch contains modifications only for plots, etc. It has major bugs and other problems. Only use this branch to reproduce the exact same plots (style, labels, etc). Otherwise use master branch.

This branch requires: https://github.com/garrettj403/SciencePlots


# project setup
Some useful commands / hints to set up the project with inside a docker container.


# setup container
create volumes and docker container
```
sudo su
mkdir /home/ubuntu/markus /mnt/hot-swap/markus /mnt/hot-swap/markus/datasets
docker run -it --gpus all --shm-size 50G -v /mnt/hot-swap/markus:/root/markus -v /home/ubuntu/markus:/root/markus/datasets --name markus ubuntu:21.10 /bin/bash
```

# setup project (inside container)
set env variables: $WANDB_API_KEY
```
apt update && DEBIAN_FRONTEND=noninteractive apt install iproute2 nano task-spooler git wget python3 python3-pip python3-opencv iputils-ping screen htop psmisc tree -y && pip install pipenv
cd /root/markus
git clone https://github.com/Glutamat42/ma_project_uncertainty ma_uncertainty
echo "cd /root/markus/ma_uncertainty" >> ~/.bashrc
echo "pipenv shell" >> ~/.bashrc
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc
mkdir datasets

# download required datasets to datasets/  (wget -c URL.tgz -O - | tar -xz -C datasets/)

# download models if required

cd /root/markus/ma_uncertainty

# create default config configs/default.ini

pipenv install
pipenv shell

# test (inside container), interrupt training if its working fine (2 gpus)
WANDB_MODE=offline python -m torch.distributed.launch --nproc_per_node 2 main.py --batch_size=128 --epochs 4 --csv_name=train_chapters.csv --csv_name_test=validation_chapters_1_2.csv --data_path=/root/markus/datasets/drive360_prepared_4_5k --dino_batch_size=90 --dino_epochs=100 --dino_lr=0.009955600488441817 --dino_min_lr=1.4e-06 --lr=0.001 --lr_scheduler=linear --min_lr=0.00091675 --mode=dino --optimizer=adamw --sampler_type=DistributedSampler --warmup_epochs=0 --weight_decay=0.001 --net_resolution 224 224 --num_workers 7 --name testrun
rm -r ../v5_outputs/testrun/
```

limit gpus for docker container: `--gpus '"device=0,1,2,3"'`

workaround apt update error (old docker version with new ubuntu version): `docker run -it --gpus all --shm-size 50G -v /home/student/workspace/markus:/root/markus --name markus ubuntu:20.04 /bin/bash`

# useful commands
Limit gpus for one process (training, ...): `CUDA_VISIBLE_DEVICES=0,1`

Change master port for multiple runs on same machine. does not work with torch.distributed.launch (its overriding that variable): `MASTER_PORT=32120`

launch multi gpu: `python -m torch.distributed.launch --nproc_per_node=2 main.py --abcdef`


# checkpoints
## filename persistent saves
`checkpoint_<head|dino|backbone>_<arch>_ep<XXXX>_<uuid>.pth`

## filename checkpoints
`checkpoint_{type}_{arch}.pth`

## statedict format
```
{
    version: int  # now: 1, if not existing: old format
    type: one_of(dino|head|backbone)
    epoch: int
    uuid: uuid.v4  # model uuid to identify this model
    arch: str
    optimizer: state_dict
    loss_func: state_dict
    config: dict
    
    student: state_dict  # if type == dino
    teacher: state_dict  # if type == dino
    
    backbone: state_dict  # if type == backbone
    
    head: state_dict  # if type == head
    uuid_backbone: uuid.v4  # only if type == head
}
```
### changes storage format 3
instead of head and backbone there is only supervised. a supervised save always contains a head but not always a backbone. 

optimizer and loss_func are now optional (they required too much disk space). If they are not stored resuming training is not possible. No problem for supervised because training is relatively fast.

new property: `arch_head` contains arch of head (because for supervised `arch` contains the arch of backbone)

### changes format 4
guaranteed to contain `metrics` dict (version 3 might contain it - latest checkpoints have it already).

### changes format 5
fixed wrong storage format of optimizer (fix for older buggy formats is included in loading logic)

# multinode training
- NCCL requires direct access to the network adapter. Forwarding/Bridging a port is not enough (--net=host)
- all gpus have to be used (can be limited with CUDA_VISIBLE_DEVICES=0)
- correct network adapter has to be set (auto detect didn’t work for me) NCCL_SOCKET_IFNAME=eno1
- Useful debug log requires NCCL log: NCCL_DEBUG=INFO
- Requires very high bandwidth between nodes. For one 1080 per node around 25GBit/s.
Example cmds:
```bash
NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=10.11.12.13 --master_port=29400 main.py
NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=10.11.12.13 --master_port=29400 main.py
```

