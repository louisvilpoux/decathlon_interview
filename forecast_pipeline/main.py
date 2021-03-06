from read import Reader
from transform import Transformer
from train import Trainer
from predict import Predicter
import pandas as pd
import logger
import os
import sys
pd.options.mode.chained_assignment = None


def main():
    try:
        logger.logger.info("Read data")
        reader = Reader(folder, logger)
        logger.logger.info("Transfrom data")
        transformer = Transformer(reader.df_train, reader.df_test, 62, logger)
        logger.logger.info("Preprocess data")
        df_train, df_train_model, df_validate_model, df_test,\
            df_businessunit_repr, df_department_repr, \
            df_week_department_repr, threshold_date = transformer.preprocess()
        logger.logger.info("Train model")
        trainer = Trainer(df_train_model, df_validate_model,
                          threshold_date, logger)
        model, performance = trainer.train()
        df_performance = pd.DataFrame([performance], columns=['r2_train',
                                      'r2_test', 'mae_train', 'mae_test'])
        df_performance.to_csv(os.path.join(folder, "performance.csv"))
        logger.logger.info("Prediction")
        predicter = Predicter(model, df_train, df_test, df_businessunit_repr,
                              df_department_repr, df_week_department_repr,
                              logger)
        output = predicter.predict()
        output.to_csv(os.path.join(folder, "prediction.csv"))
        logger.logger.info("End of program")
    except Exception as e:
        logger.logger.error(f"Error occured during main : {e}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        folder = sys.argv[1]
        main()
    else:
        logger.logger.error(f"Command line : python main.py folder.\
                              End of the program.")
        exit()
