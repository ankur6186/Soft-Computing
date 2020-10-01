import csv 
import math
import random
from k_fold import fold
from naive_iris import scale

def shuffle(array):
	for i in range(len(array)-1,0,-1):
		j=random.randint(0,i)
		array[i],array[j]=array[j],array[i]
	return array

def initialize_population(n,attributes):
	cromosome=[]
	for i in range(n):
		c=[]
		for j in range(len(attributes)-1):
			c.append(random.randint(0,1))
		cromosome.append(c)
	return cromosome

def train(data,attr,test_set):
	#print(len(data),"++++++++++++++-----------------")
	#print(data)
	nyes=0
	nno=0
	for i in range(len(data)):
		#print(data[i][0])
		if data[i][0]=="Iris-setosa":
			nyes+=1
		if data[i][0]=="Iris-versicolor":
			nno+=1
	#print(nyes,"=======",nno)
	proYes=float(nyes)/(nyes+nno)
	proNo=float(nno)/(nyes+nno)

	p1yes=[0]*(attr-1)
	p1no=[0]*(attr-1)
	p0no=[0]*(attr-1)
	p0yes=[0]*(attr-1)
	for j in range(len(data)):
		for i in range(1,attr):
			#print(data[j][i])
			if data[j][i]=='1' and data[j][0]=="Iris-setosa":
				p1yes[i]+=1
			if data[j][i]=='1' and data[j][0]=="Iris-versicolor":
				p1no[i]+=1
			if data[j][i]=='0' and data[j][0]=="Iris-setosa":
				p0yes[i]+=1
			if data[j][i]=='0' and data[j][0]=="Iris-versicolor":
				p0no[i]+=1
		
	for i in range(len(p1yes)):
		p1no[i]=p1no[i]/float(nno)
		p1yes[i]=p1yes[i]/float(nyes)
		p0no[i]=p0no[i]/float(nno)
		p0yes[i]=p0yes[i]/float(nyes)

	acc=0
	for j in range(len(test_set)):
		yes_p=proYes
		no_p=proNo
		for i in range(1,attr):
			if test_set[j][i]=='0':
				yes_p*=p0yes[i]
				no_p*=p0no[i]
			elif test_set[j][i]=='1':
				yes_p*=p1yes[i]
				no_p*=p1no[i]

		if yes_p>no_p:
			max_prob='Iris-setosa'
		else:
			max_prob='Iris-versicolor'
		if test_set[j][0]==max_prob:
			acc+=1

	result=float(acc)/len(test_set)
	result*=100
	return result	

def naive_bayes(k,rows,attributes):
	accuracy=[]
	for i in range(1,k+1):
		after_fold=fold(rows,i,k)
		train_set=after_fold[0]
		test_set=after_fold[1]
		#print(len(train_set),"------------------")
		acc=train(train_set,attributes,test_set)
		accuracy.append(acc)	
	summ=0
	for i in accuracy:
		summ+=i
	return (summ/k)

def fitness_evaluation(dataset,cromosome,attributes):
	fit=[]
	k=-1
	for cromo in cromosome:
		new_dataset=[]
		for row in dataset:
			attr=0
			new_row=[]
			for i in range(1,len(cromo)+1):
				if(cromo[i-1]==1):
					new_row.append(row[i])
					attr+=1
			new_row.insert(0,row[0])
			new_dataset.append(new_row)
		#print(new_dataset)
		fit_value=naive_bayes(10,new_dataset,attr)
		#accuracy=sum(scores[0])/float(len(scores[0]))
		k+=1
		fit.append([fit_value,k])
	return fit

def selection(fit_func,cromo):
	for i in range(len(fit_func)):
		j=fit_func[i][1]
		temp=cromo[i]
		cromo[i]=cromo[j]
		cromo[j]=temp
	return cromo

def crossover(cromo,rate):
	n_crossover=int(rate*len(cromo)*(0.01))
	copy_cromo=list(cromo)
	cross=list()
	for i in range(n_crossover):
		index=random.randrange(len(copy_cromo))
		temp=copy_cromo.pop(index)
		cross.append([cromo.index(temp),temp])
	
	for i in range(len(cross)-1):
		point=random.randint(0,len(cromo[0])-2)
		c1=cross[i][1]
		c2=cross[i+1][1]
		
		for j in range(point,len(c1)):
			c1[j]=c2[j]
		cross[i][1]=c1
	
	c1=cross[-1][1]
	c2=cross[0][1]
	
	for k in range(point,len(c1)):
		c1[k]=c2[k]
	cross[-1][1]=c1
	
	for i in range(len(cross)):
		cromo[cross[i][0]]=cross[i][1]
	return cromo

def mutation(cromo,rate):
	n_mutation=int(rate*len(cromo)*(0.01))
	copy_cromo=list(cromo)
	mutation=list()
	for i in range(n_mutation):
		index=random.randrange(len(copy_cromo))
		temp=copy_cromo.pop(index)
		mutation.append([cromo.index(temp),temp])
	for  j in range(n_mutation):
		m=mutation[j]
		position=random.randrange(len(m[1]))
		if(m[1][position]==1):
			m[1][position]=0
		else:
			m[1][position]=1
	return cromo

def check_sector(array,ele):
	if ele<=array[0]:
		return 1

	for i in range(1,len(array)):
		if ele<=array[i]:
			return (i+1)

def main():
	filename="IRIS.csv"
	attributes=[]
	rows=[]
	with open(filename,'r') as csvfile:
		csvreader=csv.reader(csvfile)

		attributes=next(csvreader)
		for row in csvreader:
			rows.append(row)
	R=shuffle(rows)
	Rows=scale(R)
	#print(Rows)
	k=10
	n=30
	cross_rate=25
	mut_rate=10
	Cromosome=initialize_population(n,attributes)
	maximum=[]
	for z in range(500):
		fit_func=fitness_evaluation(Rows,Cromosome,attributes)
		fit_func.sort(key=lambda x:x[1])

		maximum.append(fit_func[0][0])
		
		new_cromo=selection(fit_func,Cromosome)
		after_cross=crossover(new_cromo,cross_rate)
		after_mut=mutation(after_cross,mut_rate)
		Cromosome=after_mut
	result=fitness_evaluation(Rows,Cromosome,attributes)
	result.sort(key=lambda x:x[0],reverse=True)
	print("The Accuracy after applying GA : ",result[0][0])
	print("For Cromosome : ",Cromosome[result[0][1]])
	print("The Features are : ")
	final_cromosome=Cromosome[result[0][1]]
	for i in range(len(attributes)-1):
		if final_cromosome[i]==1:
			print(attributes[i+1])

	print("The Maximum Accuracy : ",max(maximum))

	#print(fit_func)	

if __name__ == '__main__':
	main()

