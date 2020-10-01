def fold(dataset,i,k):
	l=len(dataset)
	start_index_test=l*(i-1)//k
	end_index_test=l*i//k

	if start_index_test==0:
		start_index_train=end_index_test
		end_index_train=l
		return [dataset[start_index_train:end_index_train],dataset[start_index_test:end_index_test]]
	elif end_index_test==l:
		start_index_train=0
		end_index_train=start_index_test
		return [dataset[start_index_train:end_index_train],dataset[start_index_test:end_index_test]]
	else:
		start_index_train_first=0
		end_index_train_first=start_index_test
		start_index_train_second=end_index_test
		end_index_train_second=l
		new_dataset=[]
		for i in range(start_index_test):
			new_dataset.append(dataset[i])
		for j in range(end_index_test,l):
			new_dataset.append(dataset[j])

		return [new_dataset,dataset[start_index_test:end_index_test]]
