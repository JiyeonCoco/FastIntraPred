from config import Config
import os


class NetManager(object):
    cfg = Config("config.yml", None)

    # CURRENT MODE
    IS_GPU = cfg.IS_GPU
    TEST = cfg.TEST
    TRAINING = cfg.TRAINING
    VALIDATION = cfg.VALIDATION

    # PATH
    DATASET_PATH = cfg.DATASET_PATH
    TRAINING_PATH = os.path.join(DATASET_PATH, 'Training')
    VALIDATION_PATH = os.path.join(DATASET_PATH, 'Validation')
    TEST_PATH = os.path.join(DATASET_PATH, 'Test')
    MODEL_PATH = cfg.SAVE_PATH
    os.makedirs(MODEL_PATH, exist_ok=True)

    # PU DATA
    PU_SIZE = cfg.PU_SIZE
    QP_LIST = cfg.QP_LIST
    MODE_NUM = cfg.MODE_NUM

    # CONDITION
    PRINT_PERIOD = cfg.PRINT_PERIOD
    DATASET_NUM = cfg.DATASET_NUM
    BATCH_SIZE = cfg.BATCH_SIZE
    OBJECT_EPOCH = cfg.OBJECT_EPOCH
    LEARNING_RATE = cfg.INIT_LEARNING_RATE
    NUM_WORKER = cfg.NUM_WORKER


