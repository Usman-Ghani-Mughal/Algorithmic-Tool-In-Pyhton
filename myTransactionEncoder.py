import pandas as pd

class myTransform:
    def __init__(self):
        self._columns = []
        self._unique_values=[]
        self.data_dic = {}
        self.data_frame = pd.DataFrame()

    def fit(self, dataset):
        self._columns = dataset.columns.tolist()
        rows_list = dataset['Items'].tolist()
        items_list = [str(element) for element in rows_list]
        for item in items_list:
            ind_item = item.split(',')
            for i in ind_item:
                if i != 'nan' and i != '':
                    if i not in self._unique_values:
                        self._unique_values.append(i)
        self.data_frame = pd.DataFrame(columns=self._unique_values)
        print(self.data_frame)

    def transform(self,dataset):
        new_row = []
        i_prev = 0
        i_next = 1
        rows_list = dataset['Items'].tolist()
        items_list = [str(element) for element in rows_list]
        for i in items_list:
            post_message = False
            num_reactions = False
            num_comments = False
            num_shares = False
            num_likes = False
            for uniq in self._unique_values:
                if uniq in i:
                    if uniq == 'post_message':
                        post_message=True
                    elif uniq == 'num_reactions':
                        num_reactions=True
                    elif uniq == 'num_comments':
                        num_comments=True
                    elif uniq == 'num_shares':
                        num_shares=True
                    elif uniq == 'num_likes':
                        num_likes=True
            self.data_frame = self.data_frame.append({'post_message':post_message,'num_reactions':num_reactions, 'num_comments':num_comments, 'num_shares':num_shares,'num_likes':num_likes}, ignore_index=True)
            i_prev += 1
            i_next += 1
            new_row = []
        return self.data_frame



# df = pd.read_csv('prepared_data_set.csv')
# te = myTransform()
# te.fit(df)
# te.transform(df)

#print(len(row['Items']))