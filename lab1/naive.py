import csv
import math
from k_fold import fold

def train(Rows,attributes,test_set):
	nyes=0
	nno=0
	for i in range(len(Rows)):
		if Rows[i][0]=="Yes":
			nyes+=1
		if Rows[i][0]=="No":
			nno+=1

	p1yes=[0]*(len(attributes)-1)
	p1no=[0]*(len(attributes)-1)
	p0no=[0]*(len(attributes)-1)
	p0yes=[0]*(len(attributes)-1)
	for j in range(len(Rows)):
		for i in range(1,len(attributes)):
			if Rows[j][i]=='1' and Rows[j][0]=="Yes":
				p1yes[i-1]+=1
			if Rows[j][i]=='1' and Rows[j][0]=="No":
				p1no[i-1]+=1
			if Rows[j][i]=='0' and Rows[j][0]=="Yes":
				p0yes[i-1]+=1
			if Rows[j][i]=='0' and Rows[j][0]=="No":
				p0no[i-1]+=1
		
	for i in range(len(p1yes)):
		p1no[i]=p1no[i]/float(nno)
		p1yes[i]=p1yes[i]/float(nyes)
		p0no[i]=p0no[i]/float(nno)
		p0yes[i]=p0yes[i]/float(nyes)

	acc=0
	for j in range(len(test_set)):
		yes_p=1
		no_p=1
		for i in range(1,len(attributes)):
			if test_set[j][i]=='0':
				yes_p*=p0yes[i-1]
				no_p*=p0no[i-1]
			elif test_set[j][i]=='1':
				yes_p*=p1yes[i-1]
				no_p*=p1no[i-1]

		if yes_p>no_p:
			max_prob='Yes'
		else:
			max_prob='No'
		if test_set[j][0]==max_prob:
			acc+=1

	result=float(acc)/len(test_set)
	result*=100
	return result	

def main():
	filename="SPECT.csv"
	attributes=[]
	rows=[]
	with open(filename,'r') as csvfile:
		csvreader=csv.reader(csvfile)

		attributes=next(csvreader)
		for row in csvreader:
			rows.append(row)

	k=10
	accuracy=[]
	summ=0

	for i in range(1,k+1):
		after_fold=fold(rows,i,k)
		train_set=after_fold[0]
		test_set=after_fold[1]
		acc=train(train_set,attributes,test_set)
		accuracy.append(acc)
	print("The Accuracy for each fold is as follows : ")
	for i in accuracy:
		summ+=i
		print(math.ceil(i))
	summ=summ/k
	print("Average Accuracy : ",summ)

if __name__ == '__main__':
	main()
