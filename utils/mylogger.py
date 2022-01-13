import inspect
import logging
import os
import shutil
import sys
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from root import absolute


class TNLog(object):
    def printfNow(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def __init__(self, name, level=logging.NOTSET, backupCount=2):
        # loggers 笔
        self.__loggers = {}
        self.__backupCount = backupCount
        self.__logger_name = name
        self.__dir = absolute(f"logs/{self.__logger_name}")

    def initial_logger(self, clear=False):
        dir = self.__dir
        if clear and os.path.isdir(dir):
            shutil.rmtree(dir)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        dir_time = time.strftime('%Y-%m-%d', time.localtime())
        handlers = {
            logging.NOTSET: os.path.join(dir, 'notset_%s.txt' % dir_time),
            logging.DEBUG: os.path.join(dir, 'debug_%s.txt' % dir_time),
            logging.INFO: os.path.join(dir, 'info_%s.txt' % dir_time),
            logging.WARNING: os.path.join(dir, 'warning_%s.txt' % dir_time),
            logging.ERROR: os.path.join(dir, 'error_%s.txt' % dir_time),
            logging.CRITICAL: os.path.join(dir, 'critical_%s.txt' % dir_time),
        }
        logLevels = handlers.keys()
        # 构建不同的处理器
        for level in logLevels:
            path = os.path.abspath(handlers[level])
            # handlers[level] = TimedRotatingFileHandler(
            #     path, "midnight", backupCount=self.__backupCount, encoding='utf-8')
            handlers[level] = RotatingFileHandler(
                path,
                maxBytes=1024 * 100,
                backupCount=self.__backupCount,
                encoding='utf-8')
            logger = logging.getLogger(self.__logger_name + "_" + str(level))
            logger.addHandler(handlers[level])
            if level == logging.INFO or level == logging.ERROR:
                logger.addHandler(logging.StreamHandler())
            logger.setLevel(level)
            self.__loggers.update({level: logger})

    def getLogMessage(self, level, message):
        frame, filename, lineNo, functionName, code, unknowField = inspect.stack(
        )[2]
        '''日志格式：[时间] [类型] [记录代码] 信息'''

        return "[%s] [%s] [%s - %s - %s] %s" % (
            self.printfNow(), level, filename, lineNo, functionName, message)

    def info(self, message):
        message = self.getLogMessage("info", message)

        self.__loggers[logging.INFO].info(message)

    def error(self, message):
        message = self.getLogMessage("error", message)

        self.__loggers[logging.ERROR].error(message)

    def warning(self, message):
        message = self.getLogMessage("warning", message)

        self.__loggers[logging.WARNING].warning(message)

    def debug(self, message):
        message = self.getLogMessage("debug", message)

        self.__loggers[logging.DEBUG].debug(message)

    def critical(self, message):
        message = self.getLogMessage("critical", message)

        self.__loggers[logging.CRITICAL].critical(message)
