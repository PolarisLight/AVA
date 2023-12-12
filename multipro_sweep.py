from Sweep_AAM import train,sweep_config,main
import wandb
import multiprocessing
from argparse import ArgumentParser

argparser = ArgumentParser()

def run_sweep(sweep_id):
    wandb.agent(sweep_id=sweep_id, function=main, count=100)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='AAM-sweep')

    # 创建多个进程
    processes = []
    for _ in range(NUM_PROCESSES):  # NUM_PROCESSES 是您想要运行的并行进程数
        p = multiprocessing.Process(target=run_sweep, args=(sweep_id,))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()
