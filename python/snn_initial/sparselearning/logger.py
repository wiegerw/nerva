import copy
import hashlib
import logging


class Logger(object):
    def __call__(self, msg: str):
        raise NotImplementedError


class FileLogger(Logger):
    def __init__(self, args):
        self.logger = logging.getLogger()

        args_copy = copy.deepcopy(args)
        args_copy.iters = 1
        args_copy.verbose = False
        args_copy.log_interval = 1
        args_copy.seed = 0

        log_path = f'./logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def __call__(self, msg: str):
        print(msg)
        self.logger.info(msg)


# only prints to standard output
class DefaultLogger(Logger):
    def __call__(self, msg: str):
        print(msg)
