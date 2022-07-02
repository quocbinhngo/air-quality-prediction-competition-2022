TRAIN_SIZE = 0.9
SEED = 42

CAT_COL_LIST = ['wd', 'station']
NUM_COL_LIST = ['year', 'month', 'week', 'day', 'week_day', 'hour', 'PM2.5', 'PM10', 'SO2','NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

DROP_NA_COL_LIST = ["PM2.5"]
FILL_COL_LIST = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

LOG_TRANSFORM_COL_LIST = ["PM10", "SO2", "NO2", "CO", "O3", "PRES", "WSPM", "RAIN"]
ONEHOT_ENCODER_COL_LIST = ['wd', 'station']
STANDARD_SCALER_COL_LIST = ['year', 'month', 'day', 'hour', 'week', 'week_day', "PM10", "SO2", "NO2", "CO", "O3", "PRES", "WSPM", "RAIN"]
ROBUST_SCALER_COL_LIST = ['TEMP', "DEWP"]

INPUT_SIZE = 53
SEQUENCE_LENGTH = 288
BATCH_SIZE = 64
NUM_EPOCHS = 10





