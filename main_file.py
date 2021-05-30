from tkinter import *
import tkinter as tk
from tkinter.font import Font
from tkinter import filedialog
import pandas as pd
from tkinter import messagebox
import navie_bayes as nb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import kmean
import wikipedia
from PIL import ImageTk, Image
import os

class Data_Mining_tool:
    def __init__(self):
       # self.font_m = Font(family="Times New Roman", size=12)
        self.file_path = ''
        self.association_rule = ''
        self.classification_root = ''
        self.clustering_root = ''
        self.no_atributes = 0
        self.no_of_rows = 0
        self.dataframe = 0
        self.columns_names = 0
        self.list_data_box = 0
        self.list_data_box_r = 0
        self.accuracy_label_ans = 0
        self.list_data_box_real_ds = 0
        self.list_data_box_t_ds = 0
        self.RSEED = 0
        self.entry = 0
        self.answe = 0
        self.back_ground_color = '#32a8a2'

        self.root = Tk()
        self.root.geometry('600x350+400+120')
        '''changing start'''
        self.canvas = Canvas(self.root, width=600, height=350)
        image = ImageTk.PhotoImage(Image.open(os.getcwd()+'\\Images\\back_img.png'))
        self.canvas.create_image(0,0, anchor=NW, image=image)
        self.canvas.pack()
        # back_label = tk.Label(self.root, image=back_imge)
        # back_label.place(relwidth=1, relheight=1)
        #self.topframe = tk.Frame(self.canvas, width=548, height=448, highlightbackground="black", highlightcolor="black", highlightthickness=1)
        #self.topframe.pack()
        '''changing End'''
        #self.root.configure(background=self.back_ground_color)
        self.make_main_frame()
        self.root.mainloop()
        
    '''
        These Bellow Functions are used For making GUI
    '''
    def make_association_rule(self):
        self.association_rule = Toplevel()
        self.association_rule.title("Association Rule Mining")
        self.association_rule.geometry('700x600+350+50')

        main_menu = Menu(self.association_rule)
        self.association_rule.config(menu=main_menu)

        # creating Preference Menu
        preference_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Preference', menu=preference_menu)

        preference_menu.add_command(label='Change Color', command=self.clicking_on_menu_options)
        preference_menu.add_command(label='Change Fonts', command=self.clicking_on_menu_options)

        preference_menu.add_separator()

        preference_menu.add_command(label='Change Theme', command=self.clicking_on_menu_options)

        # Creating Help menu
        Help_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Help', menu=Help_menu)
        Help_menu.add_command(label='About', command=self.clicking_on_menu_options)

        btn_chose_file = Button(self.association_rule, text='Choose File', command=self.selectFile_association)
        btn_chose_file.place(x=10,y=10)


        path_label = Label(self.association_rule, text=self.file_path, bg='white', width=73, height=1)
        path_label.place(x=100,y=15)

        # Creating Algorithms
        algo_label = Label(self.association_rule, text='Algorithms', bg='gray')
        # heading_label.configure(font=self.font_m)
        algo_label.place(x=20, y=80)
        #  creating RIght Side Frame
        algo_frame = Frame(self.association_rule, width=150, height=250, bd=6, borderwidth=2, bg='white', relief='solid')
        algo_frame.place(x=20, y=120)

        btn_aprori = Button(algo_frame, text='Apriori', width=18, command=self.apply_aprori)
        btn_aprori.pack()

        btn_fpgrowth = Button(algo_frame, text='FP-Growth', width=18, command = self.apply_fpGrowth)
        btn_fpgrowth.pack()


        # Creating Data Info Table

        data_info = Frame(self.association_rule, width=50, height=30, bd=6, borderwidth=2, bg='white', relief='solid')
        data_info.place(x=420, y=50)

        label_no_atri = Label(data_info, text='No of Atributes '+str(self.no_atributes), width=18)
        label_no_atri.pack()

        label_no_rows = Label(data_info, text='No of Rows '+str(self.no_of_rows), width=18)
        label_no_rows.pack()

        # Data Display

        data_display = Frame(self.association_rule, width=500, height=300, bd=6, borderwidth=2, bg='white', relief='solid')

        scroll = Scrollbar(data_display)
        scroll.pack(side=RIGHT, fill=Y)
        self.list_data_box = Listbox(data_display, yscrollcommand=scroll.set)
        self.list_data_box.pack(side=LEFT, fill=X)
        scroll.config(command= self.list_data_box.yview)
        data_display.place(x=400, y=110)


        # rd_s_label = Label(self.association_rule, text="Real Data Set", bg='gray')
        # rd_s_label.place(x=15, y=330)
        #
        # #show real data set
        # rds_display = Frame(self.association_rule, width=250, height=50, bg="lightblue")
        # scroll_rds = Scrollbar(rds_display)
        # scroll_rds.pack(side=RIGHT, fill=Y)
        # self.list_data_box_real_ds = Listbox(rds_display, yscrollcommand=scroll_rds.set)
        # self.list_data_box_real_ds.pack(side=LEFT, fill=X)
        # scroll_rds.config(command=self.list_data_box_real_ds.yview)
        # rds_display.place(x=10, y=370)
        #
        td_s_label = Label(self.association_rule, text="Transform Data Set", bg='gray')
        td_s_label.place(x=15, y=330)

        # show transdorm data set
        tds_display = Frame(self.association_rule, width=250, height=50, bg="lightblue")
        scroll_tds = Scrollbar(tds_display)
        scroll_tds.pack(side=RIGHT, fill=Y)
        self.list_data_box_t_ds = Listbox(tds_display, yscrollcommand=scroll_tds.set)
        self.list_data_box_t_ds.pack(side=LEFT, fill=X)
        scroll_tds.config(command=self.list_data_box_t_ds.yview)
        tds_display.place(x=10, y=370)

        result_label = Label(self.association_rule, text="Frequent Items Set", bg='gray')
        result_label.place(x=250, y=330)

        fds_display = Frame(self.association_rule, width=200, height=50, bg="lightblue")
        scroll_fds = Scrollbar(fds_display)
        scroll_fds.pack(side=RIGHT, fill=Y)
        self.list_data_box_f_ds = Listbox(fds_display, yscrollcommand=scroll_fds.set)
        self.list_data_box_f_ds.pack(side=LEFT, fill=X)
        scroll_fds.config(command=self.list_data_box_f_ds.yview)
        fds_display.place(x=250, y=370)

    def make_classification(self):
        self.classification_root = Toplevel()
        self.classification_root.title("Classification")
        self.classification_root.geometry('700x600+350+50')

        _frame = tk.Frame(self.classification_root, width=700, height=600, bd=6, borderwidth=2,bg='#00070F', relief='solid')
        _frame.pack()

        main_menu = Menu(self.classification_root)
        self.classification_root.config(menu=main_menu)

        # creating Preference Menu
        preference_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Preference', menu=preference_menu)

        preference_menu.add_command(label='Change Color', command=self.clicking_on_menu_options)
        preference_menu.add_command(label='Change Fonts', command=self.clicking_on_menu_options)

        preference_menu.add_separator()

        preference_menu.add_command(label='Change Theme', command=self.clicking_on_menu_options)

        # Creating Help menu
        Help_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Help', menu=Help_menu)
        Help_menu.add_command(label='About', command=self.clicking_on_menu_options)
        h_font = Font(family='Times New Roman', size=12)
        #fg='#00AFCA', font=uni_font, bg='#000912')

        btn_chose_file = Button(self.classification_root, font=h_font, fg='#00AFCA' , bg='#000912' ,text='Choose File', command=self.selectFile_classification)
        btn_chose_file.place(x=10, y=10)

        path_label = Label(self.classification_root, text=self.file_path, width=73, height=1, fg='#00AFCA' , bg='#000912')
        path_label.place(x=100, y=15)

        # Creating Algorithms
        algo_label = Label(self.classification_root, font=h_font, text='Select An Algorithms', fg='#00AFCA' , bg='#000912', width=25,borderwidth=2,relief="groove")
        # heading_label.configure(font=self.font_m)
        algo_label.place(x=23, y=80)
        #  creating RIght Side Frame
        algo_frame = Frame(self.classification_root, width=150, height=250, bd=6, borderwidth=2,relief='solid')
        algo_frame.place(x=20, y=120)

        btn_navie = Button(algo_frame, text='Naive Bayesian Classification', fg='#00AFCA' , bg='#000912', width=25, command=self.apply_Navie_B)
        btn_navie.pack()

        btn_Decision_Trees = Button(algo_frame, text='Decision Trees', fg='#00AFCA' , bg='#000912', width=25, command=self.apply_decissiontree)
        btn_Decision_Trees.pack()


        btn_svm = Button(algo_frame, text='Support Vector Machine', fg='#00AFCA' , bg='#000912', width=25, command=self.apply_SVC)
        btn_svm.pack()

        btn_rf = Button(algo_frame, text='Random Forest', fg='#00AFCA' , bg='#000912', width=25, command=self.apply_random_forest)
        btn_rf.pack()
        # Creating Data Info Table

        data_info = Frame(self.classification_root, width=50, height=50, bg='#000912',borderwidth=2,relief="groove")
        data_info.place(x=420, y=50)

        label_no_atri = Label(data_info, fg='#00AFCA' , bg='#000912', text='No of Atributes ' + str(self.no_atributes), width=18)
        label_no_atri.pack()

        label_no_rows = Label(data_info, fg='#00AFCA' , bg='#000912', text='No of Rows ' + str(self.no_of_rows), width=18)
        label_no_rows.pack()

        # Data Display

        data_display = Frame(self.classification_root,height=500, bd=6, borderwidth=1, bg='#000912',relief='solid')

        scrolly = Scrollbar(data_display,width=15)
        scrollx = Scrollbar(data_display,orient="horizontal")

        scrollx.pack(side=BOTTOM, fill=X)
        scrolly.pack(side=RIGHT, fill=Y)

        self.list_data_box = Listbox(data_display, yscrollcommand=scrolly.set, xscrollcommand=scrollx.set, fg='#00AFCA' , bg='#000912',width=40)
        self.list_data_box.pack(side=LEFT)
        scrolly.config(command=self.list_data_box.yview)
        scrollx.config(command=self.list_data_box.xview)
        data_display.place(x=300, y=120)

        result_label = Label(self.classification_root, fg='#00AFCA' , bg='#000912', text="Results", font=h_font,borderwidth=2,relief="groove")
        result_label.place(x=60, y=290)

        result_expect = Label(self.classification_root, fg='#00AFCA' , bg='#000912', text="Expect", font=h_font,borderwidth=2,relief="groove")
        result_expect.place(x=15, y=320)

        result_pre = Label(self.classification_root, fg='#00AFCA' , bg='#000912', text="Predict", font=h_font,borderwidth=2,relief="groove")
        result_pre.place(x=107, y=320)


        result_display = Frame(self.classification_root, width=300, height=100, bg="lightblue")

        scrolly_r = Scrollbar(result_display)
        scrollx_r = Scrollbar(result_display,orient="horizontal")

        scrolly_r.pack(side=RIGHT, fill=Y)
        scrollx_r.pack(side=BOTTOM, fill=X)

        self.list_data_box_r = Listbox(result_display, yscrollcommand=scrolly_r.set,xscrollcommand=scrollx_r.set,fg='#00AFCA' , bg='#000912',width=19)
        self.list_data_box_r.pack(side=LEFT)

        scrolly_r.config(command=self.list_data_box_r.yview)
        scrollx_r.config(command=self.list_data_box_r.xview)

        result_display.place(x=10, y=360)

        accuracy_label = Label(self.classification_root, text="Accuracy", fg='#00AFCA' , bg='#000912',borderwidth=2,relief="groove",width=10)
        accuracy_label.place(x=300, y=370)
        self.accuracy_label_ans = Label(self.classification_root, text="Accuracy will show here", fg='#00AFCA' , bg='#000912', width=30,borderwidth=2,relief="groove")
        self.accuracy_label_ans.place(x=400, y=370)

        botum_frame = tk.Frame(self.classification_root, width=380, height=180, bd=6, borderwidth=2, bg='#000000',relief='solid')
        btm_img = Canvas(botum_frame,width=380,height=180)
        image = ImageTk.PhotoImage(Image.open('P:\\study\\programs\\python programs\\Algorithmic_tool\\venv\\a.png'))
        btm_img.create_image(0, 0, anchor=NW, image=image)
        btm_img.pack()
        botum_frame.place(x=300, y=405)

    def make_clustering(self):
        self.clustering_root = Toplevel()
        self.clustering_root.title("Clustering")
        self.clustering_root.geometry('700x600+350+50')

        main_menu = Menu(self.clustering_root)
        self.clustering_root.config(menu=main_menu)

        # creating Preference Menu
        preference_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Preference', menu=preference_menu)

        preference_menu.add_command(label='Change Color', command=self.clicking_on_menu_options)
        preference_menu.add_command(label='Change Fonts', command=self.clicking_on_menu_options)

        preference_menu.add_separator()

        preference_menu.add_command(label='Change Theme', command=self.clicking_on_menu_options)

        # Creating Help menu
        Help_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Help', menu=Help_menu)
        Help_menu.add_command(label='About', command=self.clicking_on_menu_options)

        btn_chose_file = Button(self.clustering_root, text='Choose File', command=self.selectFile_clustering)
        btn_chose_file.place(x=10, y=10)

        path_label = Label(self.clustering_root, text=self.file_path, bg='white', width=73, height=1)
        path_label.place(x=100, y=15)

        # Creating Algorithms
        algo_label = Label(self.clustering_root, text='Algorithms', bg='gray')
        # heading_label.configure(font=self.font_m)
        algo_label.place(x=20, y=80)
        #  creating RIght Side Frame
        algo_frame = Frame(self.clustering_root, width=150, height=250, bd=6, borderwidth=2, bg='white',
                           relief='solid')
        algo_frame.place(x=20, y=120)


        btn_kNN = Button(algo_frame, text='K-Nearest Neighbor', width=25, command=self.apply_KNN)
        btn_kNN.pack()

        btn_msc = Button(algo_frame, text='Mean-Shift Clustering', width=25)
        btn_msc.pack()

        btn_ahc = Button(algo_frame, text='Hierarchical Clustering', width=25)
        btn_ahc.pack()


        # Creating Data Info Table

        data_info = Frame(self.clustering_root, width=50, height=50, bd=6, borderwidth=2, bg='white',relief='solid')
        data_info.place(x=420, y=60)

        label_no_atri = Label(data_info, text='No of Atributes ' + str(self.no_atributes), width=18)
        label_no_atri.pack()

        label_no_rows = Label(data_info, text='No of Rows ' + str(self.no_of_rows), width=18)
        label_no_rows.pack()

        # Data Display

        data_display = Frame(self.clustering_root, width=400, height=300, bd=6, borderwidth=2, bg='white',
                             relief='solid')

        scroll = Scrollbar(data_display)
        scroll.pack(side=RIGHT, fill=Y)
        self.list_data_box = Listbox(data_display, yscrollcommand=scroll.set)
        self.list_data_box.pack(side=LEFT)
        scroll.config(command=self.list_data_box.yview)
        data_display.place(x=400, y=130)

        result_label = Label(self.clustering_root, text="Results", bg='gray')
        result_label.place(x=15, y=370)

        result_display = Frame(self.clustering_root, width=670, height=170, bg="lightblue")
        result_display.place(x=10, y=400)

    def make_searchLearn(self):
        master_sl = Tk()

        topframe_sl = Frame(master_sl)
        self.entry = Entry(topframe_sl)
        self.entry.pack()
        btn_sl = Button(topframe_sl, text='Search', command=self.get_info)
        btn_sl.pack()

        topframe_sl.pack(side=TOP)

        btomframe_sl = Frame(master_sl)

        scroll_sl = Scrollbar(btomframe_sl)
        scroll_sl.pack(side=RIGHT, fill=Y)
        self.answe = Text(btomframe_sl, width=120, height=25, yscrollcommand=scroll_sl.set, wrap=WORD)
        scroll_sl.config(command=self.answe.yview)
        self.answe.pack()
        btomframe_sl.pack()

        master_sl.mainloop()

    def make_main_frame(self):
        main_menu = Menu(self.root)
        self.root.config(menu=main_menu)

        # creating Preference Menu
        preference_menu = Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label='Preference', menu=preference_menu)
        # creating Inner Buttons/commands for Preference
        preference_menu.add_command(label='Change Color', command=self.clicking_on_menu_options)
        preference_menu.add_command(label='Change Fonts', command=self.clicking_on_menu_options)
        # Adding Seprater between Buttons/commands for Preference
        preference_menu.add_separator()
        # Adding more Buttons/commands for Preference
        preference_menu.add_command(label='Change Theme', command=self.clicking_on_menu_options)

        # Creating Help menu
        Help_menu = Menu(main_menu, tearoff=False)
        # creating Inner Buttons/commands for Help
        main_menu.add_cascade(label='Help', menu=Help_menu)
        Help_menu.add_command(label='About', command=self.clicking_on_menu_options)

        # Creating Catogories
        '''
        #$#$#$%#$%#$%@#%@#%^@#%@#$%@#$%@#$%@#$%@#
        '''
        heading_font = Font(family='Times New Roman', size=20, weight='bold')
        '''changing start'''
        heading_label = Label(self.canvas, text='Categories', font=heading_font, bg='black', fg='#00AFCA', pady=5, relief="groove")
        #heading_label.configure(font=self.font_m)
        heading_label.place(x=423, y=10)
        #  creating RIght Side Frame
        r_s_frame = tk.Frame(self.canvas, width=150, height=250, bd=6, borderwidth=2, bg='white' , relief='solid')
        r_s_frame.place(x=410, y=70)

        btn_assocation_rule = Button(r_s_frame, bg='#000912' , fg='#00AFCA' ,text='Assocation Rule Mining', width=18, command=self.make_association_rule)
        btn_assocation_rule.pack()

        btn_classification = Button(r_s_frame,  bg='#000912' , fg='#00AFCA' , text='Classification', width=18, command=self.make_classification)
        btn_classification.pack()

        btn_clustring = Button(r_s_frame,  bg='#000912' , fg='#00AFCA' , text='Clustering', width=18, command=self.make_clustering)
        btn_clustring.pack()
        btn_search_learn = Button(r_s_frame,  bg='#000912' , fg='#00AFCA' , text='Search Learn', width=18, command=self.make_searchLearn)
        btn_search_learn.pack()

        # create Image
       #path = ""
        # img = PhotoImage(file="logo1.png")
        # label_img = Label(self.root, image=img, width=100, height=100)
        # label_img.place(x=100, y=100)

        # creating bottom Left Info

        self.botumframe = tk.Frame(self.canvas, width=220, height=114, bg='#000912' , highlightbackground="black", highlightthickness=2)
        self.botumframe.place(x=375, y=230)
        info_font = Font(family='Times New Roman', size=8)
        lab1 = Label(self.botumframe, text='Data Mining Tool', width=16, height=1 ,font=info_font, fg='#00AFCA', bg='#000912')
        lab1.place(x=2, y=8)
        lab2 = Label(self.botumframe, text='Vesion 1.0' , fg='#00AFCA',font=info_font, width=16, height=1, bg='#000912')
        lab2.place(x=2, y=30)
        lab3 = Label(self.botumframe, text='(c)2019 - 2021' , fg='#00AFCA',font=info_font, width=16, height=1, bg='#000912')
        lab3.place(x=2, y=52)

        lab5 = Label(self.botumframe, text='Usman Ghani Mughal' , fg='#00AFCA',font=info_font, width=17, height=1, bg='#000912')
        lab5.place(x=107, y=8)
        lab6 = Label(self.botumframe, text='SP17-BCS-087' , fg='#00AFCA',font=info_font, width=17, height=1, bg='#000912')
        lab6.place(x=107, y=30)
        lab7 = Label(self.botumframe, text='BCS-6D', fg='#00AFCA', font=info_font, width=17, height=1, bg='#000912')
        lab7.place(x=107, y=52)

        uni_font = Font(family='Times New Roman', size=9)
        lab4 = Label(self.botumframe, text='Comsats University Islamabad (wah)', fg='#00AFCA', font=uni_font, width=30, height=1, bg='#000912')
        lab4.place(x=2, y=85)
        #, bg='#000912'

    '''
        These Bellow Functions are Used for Getting Read and apply check if data is properly Loaded
    '''
    def apply_Navie_B(self):
        try:
            if len(self.dataframe.columns) == 0:
                messagebox.showerror('Error', 'No Data is Selected')
            else:
                self.naviebayis_implementaion()
        except:
            messagebox.showerror('Error', 'There is some Error try again!')

    def apply_fpGrowth(self):
        try:
            if len(self.dataframe.columns) == 0:
                messagebox.showerror('Error', 'No Data is Selected')
            else:
                self.fpgrowth_implementaion()
        except:
            messagebox.showerror('Error', 'There is some Error try again!')

    def apply_SVC(self):
        try:
            if self.dataframe.column == 0:
                messagebox.showerror('Error', 'No Data is Selected')
        except:
            self.SVC_implementaion()

    def apply_decissiontree(self):
        try:
            if len(self.dataframe.columns) == 0:
                messagebox.showerror('Error', 'No Data is Selected')
            else:
                self.decisontree_implementation()
        except:
            messagebox.showerror('Error', ' There is Some Error Try Again!')

    def apply_aprori(self):
        try:
            if self.dataframe.column == 0:
                messagebox.showerror('Error', 'No Data is Selected')
        except:
            te = TransactionEncoder()
            te_ary = te.fit(self.dataframe).transform(self.dataframe)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            temp_string = ''
            for index, row in df.iterrows():
                for columnname in df.columns:
                    temp_string = temp_string + str(row[columnname]) + ', '
                self.list_data_box_t_ds.insert(END, temp_string)
                temp_string = ''

            frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
            temp_string = ''
            for index, row in frequent_itemsets.iterrows():
                for columnname in frequent_itemsets.columns:
                    temp_string = temp_string + str(row[columnname]) + ', '
                self.list_data_box_f_ds.insert(END, temp_string)
                temp_string = ''

    def apply_random_forest(self):
        try:
            if len(self.dataframe.columns) == 0:
                messagebox.showerror('Error', 'No Data is Selected')
            else:
                self.random_forest_implementation()
        except:
            messagebox.showerror('Error', ' There is Some Error Try Again!')

    def apply_KNN(self):
        try:
            if len(self.dataframe.columns) == 0:
                messagebox.showerror('Error', 'No Data is Selected')
            else:
                self.knn_implementaion()
        except:
            messagebox.showerror('Error', ' There is Some Error Try Again!')


    '''
        These Bellow Functions are used fro Applying Algorithms
    '''
    def fpgrowth_implementaion():
        patterns = pyfpgrowth.find_frequent_patterns(self.dataframe , 0.01)
        rules = pyfpgrowth.generate_association_rules(patterns, 0.8)

    def naviebayis_implementaion(self):
        nav_by = nb.navieBayes()
        train_x, train_y, test_x, test_y = nav_by.split_train_test(self.dataframe)
        nav_by.fit(train_x, train_y)
        # print(nav_by.prdict([5,35,8,45,91330,4,1.00,2,0,0,0,0,0]))
        y_expect = test_y
        y_predict = nav_by.predict_onlist(test_x)
        accuracy = nav_by.find_accuracy(y_expect, y_predict)
        self.list_data_box_r.delete(0, END)
        self.accuracy_label_ans['text'] = str(accuracy)
        for (ex, pr) in zip(y_expect, y_predict):
            self.list_data_box_r.insert(END, "  "+str(ex) + "                  " + str(pr))


    def SVC_implementaion(self):
        x = self.dataframe.iloc[:, :-1]
        y = self.dataframe.iloc[:, -1].tolist()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        self.list_data_box_r.delete(0, END)
        self.accuracy_label_ans['text'] = str(accuracy)
        for (ex, pr) in zip(y_test, y_pred):
            self.list_data_box_r.insert(END, str(ex) + "          ,          " + str(pr))

    def decisontree_implementation(self):
        x = self.dataframe.iloc[:, :-1]
        y = self.dataframe.iloc[:, -1].tolist()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        self.list_data_box_r.delete(0, END)
        self.accuracy_label_ans['text'] = str(accuracy)
        for (ex, pr) in zip(y_test, y_pred):
            self.list_data_box_r.insert(END, str(ex) + "          ,          " + str(pr))

    def random_forest_implementation(self):
        self.RSEED = 50
        x = self.dataframe.iloc[:, :-1]
        y = self.dataframe.iloc[:, -1].tolist()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # Feature Scaling

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        self.list_data_box_r.delete(0, END)
        self.accuracy_label_ans['text'] = str(accuracy)
        for (ex, pr) in zip(y_test, y_pred):
            self.list_data_box_r.insert(END, str(ex) + "          ,          " + str(pr))

    def knn_implementaion(self):
        obj = Kmeans()
        obj.fit(df, 2)

    '''
        These Bellow Functions are used for Selecting a Data Set
    '''
    def selectFile_association(self):
        self.file_path = filedialog.askopenfilename()
        print(self.file_path)
        self.dataframe = pd.read_csv(self.file_path)
        self.set_no_rows()
        self.set_no_atributes()
        self.make_association_rule()
        self.show_data()

    def selectFile_classification(self):
        self.file_path = filedialog.askopenfilename()
        print(self.file_path)
        self.dataframe = pd.read_csv(self.file_path)
        self.set_no_rows()
        self.set_no_atributes()
        self.make_classification()
        self.show_data()

    def selectFile_clustering(self):
        self.file_path = filedialog.askopenfilename()
        print(self.file_path)
        self.dataframe = pd.read_csv(self.file_path)
        self.set_no_rows()
        self.set_no_atributes()
        self.make_clustering()
        self.show_data()

    '''
        These Functions are some Helping Functions
    '''
    def set_no_rows(self):
        self.no_of_rows = len(self.dataframe.index)

    def set_no_atributes(self):
        self.no_atributes = len(self.dataframe.columns)
        self.columns_names = self.dataframe.columns

    def show_data(self):
        temp_string = ''
        for index, row in self.dataframe.iterrows():
            for columnname in self.columns_names:
                temp_string = temp_string+str(row[columnname])+', '
            self.list_data_box.insert(END, temp_string)
            temp_string = ''

    def get_info(self):
        try:
            entry_value = self.entry.get()
            ansvale = wikipedia.summary(entry_value)
            self.answe.delete(1.0, END)

            self.answe.insert(INSERT, ansvale)
        except:
            self.answe.delete(1.0, END)
            self.answe.insert(INSERT, "Please check your input or internet conection")

    def clicking_on_menu_options(self):
        print("Yes Yes Ok")

    def inner_file_btn(self):
        print("Inner Btn is CLicked")


obj = Data_Mining_tool()