import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'w')
	# write()函数这样写，每调用一次就写到记录文件中，不需要等待程序运行结束。
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
	    pass