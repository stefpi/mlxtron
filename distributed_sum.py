import mlx.core as mx

data = [1, 2, 3, 4]

def main():
    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()
    print(f"Process {rank + 1}/{world_size} initialized.", flush=True)

    data_seg_len = len(data)//world_size
    data_rank = data[rank*data_seg_len:(rank + 1)*data_seg_len]
    sum = 0
    for x in data_rank:
        sum += x
    print(f"Process {rank} computed data sum to be {sum}", flush=True)

    total_sum = mx.distributed.all_sum(sum)
    print(f"Data sum total = {total_sum}", flush=True)

if __name__ == "__main__":
    main()