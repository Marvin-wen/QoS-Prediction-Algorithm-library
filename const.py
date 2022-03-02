import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIRNAME = "data"  # 数据集放在根目录的data文件夹下

# 确定文件名正确
RT_MATRIX_NAME = "rtMatrix.txt"
TP_MATRIX_NAME = "tpMatrix.txt"
USERS_NAME = "userlist.txt"
WSLIST_NAME = "wslist.txt"

DATASET_DIR = os.path.join(BASE_DIR, DATASET_DIRNAME)

RT_MATRIX_DIR = os.path.join(DATASET_DIR, RT_MATRIX_NAME)
TP_MATRIX_DIR = os.path.join(DATASET_DIR, TP_MATRIX_NAME)
USER_DIR = os.path.join(DATASET_DIR, USERS_NAME)
WS_DIR = os.path.join(DATASET_DIR, WSLIST_NAME)

__all__ = ["RT_MATRIX_DIR", "TP_MATRIX_DIR", "USER_DIR", "WS_DIR"]
