import csv
import matplotlib.pyplot as plt

from comp.conf import export_file_name
from comp.conf import labels_file_name

def evaluate(correct_genders,correct_ages,genders,ages):#returns accuracy,precision,recall,mse
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    count = 0
    #we assume male=negative and female=positive
    for short_code in genders.keys():
        if short_code not in correct_genders:
            continue
        count = count+1
        if correct_genders[short_code] == genders[short_code]:
            if correct_genders[short_code] == "male":
                TN = TN +1
            if correct_genders[short_code] == "female":
                TP = TP +1
        else:
            if correct_genders[short_code] == "male":
                FN = FN +1
            if correct_genders[short_code] == "female":
                FP = FP +1
    try:
        accuracy = (TP+TN)/count
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)

        total_se = 0
        count = 0
        for short_code in ages.keys():
            if short_code not in correct_ages:
                continue
            count = count+1
            total_se += (correct_ages[short_code]-ages[short_code])**2        

        mse = total_se/count

    except:
        print('DEVISION BY ZERO')
        return 1,1,1,1
            
            
    return accuracy,recall,precision,mse


def plot(title,x,y,name1,value1,name2,value2):
    xx = [name1,name2]
    yy = [value1,value2]

    x_pos = [i for i, _ in enumerate(xx)]

    plt.bar(x_pos, yy, color='green')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)

    plt.xticks(x_pos, xx)

    #plt.show()
    plt.savefig(title+".png")#or pdf


def compare():    
    labels_file = open(labels_file_name, 'r')
    labels_reader = csv.reader(labels_file, delimiter=',')
    next(labels_reader)#remove header row

    correct_genders = {}
    correct_ages = {}
    for row in labels_reader:
        correct_genders[row[0]]=row[1]
        correct_ages[row[0]] = int(row[2])

    smahesh_genders = {}
    smahesh_ages = {}

    agender_genders = {}
    agender_ages = {}


    preds_file = open(export_file_name, 'r')
    preds_reader = csv.reader(preds_file, delimiter=',')
    next(preds_reader)#remove header row

    for row in preds_reader:
        model = row[1]
        age = row[3]
        if age == '':
            age = '0'

        if model=="smahesh":
            smahesh_genders[row[0]]=row[2]
            smahesh_ages[row[0]] = float(age)
        if model=="agender":
            agender_genders[row[0]]=row[2]
            agender_ages[row[0]] = float(age)



    accuracy1,recall1,precision1,mse1 = evaluate(correct_genders,correct_ages,smahesh_genders,smahesh_ages)
    accuracy2,recall2,precision2,mse2 = evaluate(correct_genders,correct_ages,agender_genders,agender_ages)

    plot("Accuracy (gender)","Model","Accuracy","smahesh",accuracy1,"agender",accuracy2);
    plot("Recall (gender)","Model","Recall","smahesh",recall1,"agender",recall2);
    plot("Precision (gender)","Model","Precision","smahesh",precision1,"agender",precision2);
    plot("Mean Squared Error (age)","Model","MSE","smahesh",mse1,"agender",mse2);


    print("smahesh: Accuracy=%5.2f, Recall=%5.2f, Precision=%5.2f, MSE=%5.2f"%(accuracy1,recall1,precision1,mse1))
    print("agender: Accuracy=%5.2f, Recall=%5.2f, Precision=%5.2f, MSE=%5.2f"%(accuracy2,recall2,precision2,mse2))

