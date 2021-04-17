from read import Reader
import logger
import datetime
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet
import pandas as pd

class Transformer():
    def __init__(self, df_train, df_test, duration_predict, logger):
        self.df_train = df_train
        self.df_test = df_test
        self.logger = logger
        self.duration_predict = duration_predict

    def check_columns(self):
        '''
        Function to check if dataframes have the correct column names
        Input :
            Nothing
        Return :
            check_train, check_test : True if the dataframes have the correct column names, False otherwise
        '''
        columns_train = ["day_id","but_num_business_unit","dpt_num_department","turnover"]
        check_train = all(col_df in columns_train for col_df in self.df_train.columns)
        columns_test = ["day_id","but_num_business_unit","dpt_num_department"]
        check_test = all(col_df in columns_test for col_df in self.df_test.columns)
        return check_train,check_test

    def convert_column_datetime(self, df, column_name, format="%Y-%m-%d"):
        '''
        Function to convert the column of a dataframe to a datetime column
        Input :
            df : the input dataframe
            column_name : the name of the column to convert to a datetime column
            format : the format of the date string of the column
        Return :
            df : the updated dataframe
        '''
        df[column_name] = pd.to_datetime(df[column_name], format=format)
        return df
    
    def add_year_column(self, df, date_column):
        '''
        Function to add a year column to a dataframe
        Input :
            df : the input dataframe
            date_column : the column from the dataframe to use to calculate the year information
        Return :
            df : the dataframe with a year column added
        '''
        df["year"] = df[date_column].dt.year
        return df

    def representation_businessunit_from_year(self):
        '''
        Function to obtain the mean turnover percentage of the different business units from the annual turnover
        Input :
            Nothing
        Return :
            df_businessunit_mean : dataframe of the percentages to apply for each business unit
        '''
        df_turnover_businessunit = self.df_train[["but_num_business_unit","turnover","year"]].copy()
        df_turnover_businessunit = df_turnover_businessunit.groupby(["year","but_num_business_unit"]).sum()
        df_turnover_businessunit["bu_percentage"] = df_turnover_businessunit.groupby(level=0).apply(lambda x:x / float(x.sum()))
        df_turnover_businessunit = df_turnover_businessunit.reset_index()
        df_businessunit_mean = df_turnover_businessunit.groupby("but_num_business_unit").mean().reset_index()
        df_businessunit_mean = df_businessunit_mean[["but_num_business_unit","bu_percentage"]]
        return df_businessunit_mean

    def representation_department_from_businessunit(self):
        '''
        Function to obtain the mean turnover percentage of the departments from the business units
        Input :
            Nothing
        Return :
            df_department_mean : dataframe of the percentages to apply for each department
        '''
        df_department_businessunit = self.df_train[["but_num_business_unit","dpt_num_department","turnover"]].copy()
        df_department_businessunit = df_department_businessunit.groupby(["but_num_business_unit","dpt_num_department"]).sum()
        df_department_businessunit["percentage"] = df_department_businessunit.groupby(level=0).apply(lambda x:x / float(x.sum()))
        df_department_businessunit = df_department_businessunit.reset_index()
        repres_department = list()
        department = list()
        for dep in list(df_department_businessunit["dpt_num_department"].unique()):
            filt_department = df_department_businessunit["dpt_num_department"]==dep
            df_dep = df_department_businessunit[filt_department]
            mean_val = df_dep["percentage"].mean()
            department.append(dep)
            repres_department.append(mean_val)
        columns = ["dpt_num_department", "department_percentage"]
        df_department_mean = pd.DataFrame(list(zip(department, repres_department)),columns=columns)
        return df_department_mean

    def representation_week_from_department(self):
        '''
        Function to obtain the mean weekly turnover percentage of the departments
        Input :
            Nothing
        Return :
            df_week_department_mean : dataframe of the percentages to apply for each week and departement
        '''
        df_week_department = self.df_train[["day_id","dpt_num_department","turnover"]].copy()
        df_week_department["week"] = df_week_department["day_id"].dt.isocalendar().week
        df_week_department["year"] = df_week_department["day_id"].dt.year
        df_week_year_department = df_week_department.groupby(["dpt_num_department","week","year"]).sum().reset_index()
        df_year_department = df_week_department.groupby(["dpt_num_department","year"]).sum().reset_index()[["dpt_num_department","year","turnover"]]
        df_week_department_perc = df_week_year_department.merge(df_year_department,on=["dpt_num_department","year"])
        df_week_department_perc["week_percentage"] = df_week_department_perc["turnover_x"]/df_week_department_perc["turnover_y"]
        df_week_department_perc = df_week_department_perc[["dpt_num_department","week","week_percentage"]]
        df_week_department_mean = df_week_department_perc.groupby(["dpt_num_department","week"]).mean()
        df_week_department_mean = df_week_department_mean.reset_index()[["dpt_num_department","week","week_percentage"]]
        return df_week_department_mean

    def create_dataframe_turnover_per_day(self,df):
        '''
        Function to obtain the dataframe of the turnover by day
        Input :
            df : initial dataframe
        Return :
            df_turnover_per_day : dataframe of the turnover per day
        '''
        df_turnover_per_day = df[["day_id","turnover"]].copy()
        df_turnover_per_day = df_turnover_per_day.groupby("day_id").sum().reset_index()
        return df_turnover_per_day

    def create_threshold_date(self,df):
        '''
        Function to obtain a date that was duration_predict days from the maximum date of a dataframe
        Input :
            df : initial dataframe
        Return :
            threshold_date : the corresponding date
        '''
        threshold_date = df["day_id"].max() - datetime.timedelta(days=self.duration_predict)
        return threshold_date

    def create_dataframe_train_valid(self,df,threshold_date):
        '''
        Function to obtain one dataframe of training and one dataframe for validation
        Input :
            df : initial dataframe
            threshold_date : date used for the split
        Return :
            df_train_fc : dataframe for training
            df_valid_fc : dataframe for validation
        '''
        df = df.rename(columns={"day_id":"ds","turnover":"y"})
        mask = df['ds'] < threshold_date
        df_train_fc = df[mask][['ds', 'y']]
        df_valid_fc= df[~ mask][['ds', 'y']]
        return df_train_fc,df_valid_fc

    def preprocess(self):
        '''
        Function to obtain the percentage representation of the turnover for the different granularity
        '''
        check_train, check_test = self.check_columns()
        if not check_train:
            self.logger.logger.error(f"Train data do not have the right columns. End of the program.")
            exit()
        if not check_test:
            self.logger.logger.error(f"Test data do not have the right columns. End of the program.")
            exit()
        try:
            self.df_train = self.convert_column_datetime(self.df_train,"day_id")
            self.df_test = self.convert_column_datetime(self.df_test,"day_id")
            self.df_train = self.add_year_column(self.df_train,"day_id")
            df_businessunit_repr = self.representation_businessunit_from_year()
            df_department_repr = self.representation_department_from_businessunit()
            df_week_department_repr = self.representation_week_from_department()
            df_turnover_day = self.create_dataframe_turnover_per_day(self.df_train)
            limit_date = self.create_threshold_date(df_turnover_day)
            df_train_model, df_test_model = self.create_dataframe_train_valid(df_turnover_day,limit_date)
            return df_train_model, df_test_model, self.df_test, df_businessunit_repr, df_department_repr, df_week_department_repr, limit_date
        except Exception as e:
            self.logger.logger.error(f"Error during the preprocessing : {e}")
            exit()


if __name__ == "__main__":
    reader = Reader("/home/louis/projects/perso/decathlon/test_data_scientist",logger)
    transformer = Transformer(reader.df_train,reader.df_test,62,logger)