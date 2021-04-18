from read import Reader
from transform import Transformer
from train import Trainer
import logger
import datetime
import pandas as pd


class Predicter():
    def __init__(self, model, df_train, df_test, df_businessunit_repr, df_department_repres, df_week_repres, logger):
        self.model = model
        self.df_train = df_train
        self.df_test = df_test
        self.df_businessunit_repr = df_businessunit_repr
        self.df_department_repres = df_department_repres
        self.df_week_repres = df_week_repres
        self.logger = logger

    def predict_turnover(self):
        '''
        Function to obtain the prediction from the model of the test dataframe
        Input :
            model : the model used
        Return :
            prediction_test : the dataframe of the prediction 
        '''
        future_test = pd.DataFrame({'ds': list(self.df_test["day_id"].unique())})
        prediction_test = self.model.predict(df=future_test)
        df_prediction = prediction_test[["ds","yhat"]]
        df_prediction.loc[:,"week"] = df_prediction["ds"].dt.isocalendar().week
        return df_prediction

    def predict(self):
        '''
        Function to predict the turnover by week for the different departments and business units along weeks
        Input : 
            Nothing
        Output :
            df_result : the final prediction
        '''
        try:
            df_prediction = self.predict_turnover()
            year_turnover = self.calcul_one_year_turnover(df_prediction)
            df_businessunit_year = self.apply_percentage_businessunit(year_turnover)
            df_department_businessunit_year = self.apply_percentage_department_businessunit(df_businessunit_year)
            df_week_department_businessunit_year = self.apply_percentage_department_week(df_department_businessunit_year)
            df_result = self.merge_with_original_dftest(df_week_department_businessunit_year)
            return df_result
        except Exception as e:
            self.logger.logger.error(f"Error during the prediction : {e}")
            exit()

    def calcul_one_year_turnover(self,df_prediction):
        '''
        Function to obtain the turnover made during a year that ended at the last date of the test dataframe
        Input :
            df_prediction : daframe of the result of the prediction
        Return :
            turnover_rolling_year : the turnover of the year
        '''
        date_one_year = df_prediction["ds"].max() - datetime.timedelta(days=365)
        filt_date = self.df_train["day_id"] >= date_one_year
        turnover_train_part = self.df_train[filt_date]["turnover"].sum()
        turnover_test_part = df_prediction["yhat"].sum()
        turnover_rolling_year = turnover_train_part+turnover_test_part
        return turnover_rolling_year

    def apply_percentage_businessunit(self,turnover_rolling_year):
        '''
        Function to obtain the turnover made by all of the business units
        Input :
            turnover_rolling_year : the predicted turnover of the year
        Return :
            df_result : dataframe of the turnover by business unit
        '''
        df_result = self.df_businessunit_repr.copy()
        df_result.loc[:,"turnover_bu"] = df_result["bu_percentage"]*turnover_rolling_year
        return df_result

    def apply_percentage_department_businessunit(self,df_turnover_businessunit):
        '''
        Function to obtain the turnover by department for the different business units
        Input :
            df_turnover_businessunit : dataframe of the turnover per business unit
        Return :
            df_result : dataframe of the turnover per department per business unit
        '''
        cols = list(self.df_department_repres.columns)
        df_department_repres_all = pd.DataFrame(columns=cols)
        df_department_repres_all = df_department_repres_all.append([self.df_department_repres]*322,ignore_index=True)

        cols = list(df_turnover_businessunit.columns)
        df_turnover_businessunit_all = pd.DataFrame(columns=cols)
        df_turnover_businessunit_all = df_turnover_businessunit_all.append([df_turnover_businessunit]*4,ignore_index=True)
        df_turnover_businessunit_all.sort_values(by="but_num_business_unit",inplace=True)
        df_turnover_businessunit_all = df_turnover_businessunit_all.reset_index(drop=True)

        df_result = pd.concat([df_turnover_businessunit_all,df_department_repres_all],axis=1)
        #apply percentage of department
        df_result.loc[:,"turnover_bu_dep"] = df_result["turnover_bu"]*df_result["department_percentage"]
        df_result = df_result[["but_num_business_unit","dpt_num_department","turnover_bu_dep"]]
        return df_result

    def apply_percentage_department_week(self,df_turnover_businessunit_department):
        '''
        Function to obtain the turnover by week for the different business units and department
        Input :
            df_turnover_businessunit_department : dataframe of the turnover per business unit per department
        Return :
            df_result : dataframe of the turnover per business unit per department per week
        '''
        df_result = pd.merge(df_turnover_businessunit_department, self.df_week_repres, on="dpt_num_department", how="outer")
        df_result.loc[:,"turnover_pred"] = df_result["turnover_bu_dep"]*df_result["week_percentage"]
        return df_result

    def merge_with_original_dftest(self,df_turnover_businessunit_department_week):
        '''
        Function to obtain the turnover by week for the different business units and departments from the test data
        Input :
            df_turnover_businessunit_department_week : dataframe of the turnover per business unit per department per week
        Return :
            df_result : dataframe of the turnover per business unit per department per week from the test data
        '''
        df_turnover_businessunit_department_week = df_turnover_businessunit_department_week[["but_num_business_unit","dpt_num_department","week","turnover_pred"]]
        df_turnover_businessunit_department_week.loc[:,"dpt_num_department"] = df_turnover_businessunit_department_week["dpt_num_department"].astype("int")
        df_turnover_businessunit_department_week.loc[:,"but_num_business_unit"] = df_turnover_businessunit_department_week["but_num_business_unit"].astype("int")
        self.df_test.loc[:,"week"] = self.df_test["day_id"].dt.isocalendar().week
        self.df_test.loc[:,"week"] = self.df_test["week"].astype("int")
        df_result = self.df_test.merge(df_turnover_businessunit_department_week,on=["but_num_business_unit","dpt_num_department","week"],how="left")
        return df_result


if __name__ == "__main__":
    reader = Reader("/home/louis/projects/perso/decathlon/test_data_scientist",logger)
    transformer = Transformer(reader.df_train,reader.df_test,62,logger)
    df_train, df_train_model, df_validate_model, df_test, df_businessunit_repr, \
        df_department_repr, df_week_department_repr, threshold_date = transformer.preprocess()
    trainer = Trainer(df_train_model, df_validate_model, threshold_date, logger)
    model, performance = trainer.train()
    predicter = Predicter(model, df_train, df_test, df_businessunit_repr, df_department_repr, df_week_department_repr, logger)
    output = predicter.predict()