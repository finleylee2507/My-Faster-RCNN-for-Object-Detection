import numpy as np 


def threshold_by_sampling(input,k,n):
#note: input=7x7 numpy array (dimension could change), 0<k<=1, 0<n<=1  
  (height,width)=input.shape #get the height and width of the input array 


  temp=np.copy(input) #make a copy of the input array 
  result=np.zeros((height,width)) #initialize the returned array  
  temp=temp.flatten() #convert to 1d 
  

  temp_indices=np.argsort(temp)[::-1] 
  
  temp=np.sort(temp)[::-1] #sort in descending order 
 
 
  num_of_elements=len(temp) 


  num1=int(num_of_elements*k) 

  num2=int(num1*n)
 
 
  selected_indices1=temp_indices[:num1] #grab the indices of the selected top k elements 
 
  #perform a random selection on the selected elements 
  selected_indices2=np.random.choice(selected_indices1,size=num2,replace=False)
 
  #map the result back to the original 2d array and perform thresholding 
  for index in selected_indices2:
    row=index//width 
    col=index%width 
    result[row,col]=1 

  return result 




def random_thresholding(input,n):
  ###note: input is a 7x7 numpy array, 0<n<=1
 
  (height,width)=input.shape
  result=np.zeros((height,width))
  num=height*width
  selected_num=int(num*n)
  selected_indices=np.random.choice(num,selected_num,replace=False)

  for index in selected_indices:
    row=index//width
    col=index%width 

    result[row,col]=1 

  return result
