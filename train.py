import argparse
import functools
import pandas as pd
import torch
from mvector.trainer import MVectorTrainer
from mvector.utils.utils import add_arguments, print_arguments
from visualdl import LogWriter


def check_gpu_available():
    if torch.cuda.is_available():
        print("使用GPU进行训练")
        return True
    else:
        print("无法使用GPU，将在CPU上进行训练")
        return False


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'F:/project/pytorch work/configs/ecapa_tdnn.yml',      '配置文件')
add_arg("local_rank",       int,    0,                             '多卡训练需要的参数')
add_arg("use_gpu",          bool,   True,                          '是否使用GPU训练')
add_arg('augment_conf_path',str,    'F:/project/pytorch work/configs/augmentation.json',   '数据增强的配置文件，为json格式')
add_arg('save_model_path',  str,    'model2/',                     '模型保存的路径')
add_arg('resume_model',     str,    None,                          '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                          '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args=args)

# 检查GPU是否可用
use_gpu = args.use_gpu and check_gpu_available()

# 创建LogWriter对象记录训练日志
log_writer = LogWriter(logdir='logs')

# 获取训练器
trainer = MVectorTrainer(configs=args.configs, use_gpu=use_gpu)
data = trainer.train(save_model_path=args.save_model_path,
                     resume_model=args.resume_model,
                     pretrained_model=args.pretrained_model,
                     augment_conf_path=args.augment_conf_path)
# 关闭LogWriter对象
log_writer.close()
