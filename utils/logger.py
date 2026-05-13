import logging
from datetime import datetime
import os
import sys
import pandas as pd

class logger:
    def __init__(self, args):

        self.logger = logging.getLogger('Evaluation')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicate logging when logger is re-initialized
        self.logger.handlers.clear()
        
        self.args = args

        formatter = logging.Formatter('%(message)s')

        strm_handler = logging.StreamHandler(stream=sys.stdout)
        strm_handler.setFormatter(formatter)

        now = datetime.now()
        # Windows 文件名不允许包含冒号 ":"，因此将时间戳中的 ":" 替换为 "-"
        time = now.strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(f'{args.out}', exist_ok=True)
        # 固定使用 UTF-8 写入，避免 Windows 默认编码与编辑器打开编码不一致导致乱码
        file_handler = logging.FileHandler(f'{args.out}/{time}.txt', encoding='utf-8')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(strm_handler)
        self.logger.addHandler(file_handler)

        message = f'---{args.dataset}---'
        self(message, level=1)
        self.arg_logging(args)

    def __call__(self, message, level):
        if level == 0:
            prefix = ''
        elif level == 1:
            prefix = '--->'
        else:
            prefix = '  ' * level + '└' + '>'

        self.logger.info(f'{prefix} {message}')

    def arg_logging(self, argument):
        self('Argument', level=1)
        arg_dict = vars(argument)
        for key in arg_dict.keys():
            if key == 'logger':
                pass
            else:
                self(f'{key:12s}: {arg_dict[key]}', level=2)

    def map_save(self, map):
        map_df = pd.DataFrame(map)
        map_df.to_csv(f'{self.args.out}/curriculum.csv')