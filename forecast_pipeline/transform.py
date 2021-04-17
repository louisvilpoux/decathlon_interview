from read import Reader
import logger

class Transformer():
    def __init__(self,df_train,df_test):
        self.df_train = df_train
        self.df_test = df_test





if __name__ == "__main__":
    reader = Reader("/home/louis/projects/perso/decathlon/test_data_scientist",logger)
    Transformer = Transformer(reader.df_train,reader.df_test)