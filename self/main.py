import pandas as pd

from config import *
from dataset import *
from model import *
from preprocess import *
from util import *


def main():
    print("Reading the data...")
    df_list = read_data()

    print("Preprocessing the data...")
    X_train, X_val, y_train, y_val = preprocess_data(df_list)

    # print("Preparing the dataset...")
    # data_module = prepare_dataset(X_train, X_val, y_train, y_val)
    #
    # print("Training the model...")
    # trainer, model = train_model(data_module)

    model = get_model("pm2.5-change-fill", 0, "epoch=3-step=23160.ckpt")

    predictions, labels, rmse = validate(model, X_val, y_val)
    print(rmse)

    # Save the result
    validate_df = pd.DataFrame.from_dict({"predictions": predictions,
                                          "labels": labels})
    validate_df.to_csv("./out/predict-change-fill.csv")


if __name__ == "__main__":
    main()
