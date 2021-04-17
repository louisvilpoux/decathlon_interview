from read import Reader
from transform import Transformer
import logger
import datetime
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet

class Trainer():
    def __init__(self, df_train, df_validate, threshold_date, logger):
        self.df_train = df_train
        self.df_validate = df_validate
        self.threshold_date = threshold_date
        self.logger = logger

    def train(self):
        '''
        Function to obtain a trained model
        Input :
            Nothing
        Return :
            m : the trained model
        '''
        try:
            m = Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
            m.fit(self.df_train)
            future = m.make_future_dataframe(periods=self.df_validate.shape[0], freq='W')
            forecast = m.predict(df=future)
            mask2 = forecast['ds'] < self.threshold_date
            forecast_train = forecast[mask2]
            forecast_test = forecast[~ mask2]
            print('r2 train: {}'.format(r2_score(y_true=self.df_train['y'], y_pred=forecast_train['yhat'])))
            print('r2 test: {}'.format(r2_score(y_true=self.df_validate['y'], y_pred=forecast_test['yhat'])))
            print('---')
            print('mae train: {}'.format(mean_absolute_error(y_true=self.df_train['y'], y_pred=forecast_train['yhat'])))
            print('mae test: {}'.format(mean_absolute_error(y_true=self.df_validate['y'], y_pred=forecast_test['yhat'])))
            return m
        except Exception as e:
            self.logger.logger.error(f"Error during the model training : {e}")
            exit()


if __name__ == "__main__":
    reader = Reader("/home/louis/projects/perso/decathlon/test_data_scientist",logger)
    transformer = Transformer(reader.df_train,reader.df_test,62,logger)
    df_train, df_train_model, df_validate_model, df_test, df_businessunit_repr, df_department_repr, df_week_department_repr, threshold_date = transformer.preprocess()
    trainer = Trainer(df_train_model, df_validate_model, threshold_date, logger)
    model = trainer.train()