
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn import metrics 
get_ipython().magic('matplotlib inline')


# In[2]:


INPUT_SIGNAL_TYPES= [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]
LABELS=[
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


# In[3]:


DATASET_PATH="C:\\Users\\naveen chauhan\\Desktop\\mldata\\mlp\\Human Activity Recognition\\UCI HAR Dataset\\"
TRAIN="train\\"
TEST="test\\"
X_train_signal_path=[DATASET_PATH+TRAIN+"Inertial Signals\\"+signals+"train.txt" for signals in INPUT_SIGNAL_TYPES]
#print(X_train_signal_path)
X_test_signal_path=[DATASET_PATH+TEST+"Inertial Signals\\"+signals+"test.txt" for signals in INPUT_SIGNAL_TYPES]

def Load_X(X_signal_path):
    X_signals=[]
    for signal_type_path in X_signal_path:
        file=open(signal_type_path,'r')
        X_signals.append([np.array(serie,dtype=np.float32) for serie in [row.replace('  ',' ').strip().split() for row in file]])
        file.close()
    return np.transpose(X_signals,(1,2,0))

def Load_y(y_path):
    file=open(y_path,'r')
    y_=np.array([ele for ele in [row.replace('  ',' ').strip().split() for row in file]],dtype=np.int32)
    file.close()
    return y_-1
y_train_path=DATASET_PATH+TRAIN+"y_train.txt"
y_test_path=DATASET_PATH+TEST+"y_test.txt"

y_train=Load_y(y_train_path)
y_test=Load_y(y_test_path)


# In[4]:


X_train = Load_X(X_train_signal_path)
X_test = Load_X(X_test_signal_path)


# In[5]:


print(len(X_train))
print(len(X_test))
print(len(X_train[0]))
print(len(X_test[0][0]))
X_train.shape


# In[6]:


training_data_count=len(X_train)
test_data_count=len(X_test)
n_step=len(X_train[0]) #time step per series
n_input=len(X_train[0][0])


# In[7]:


n_hidden=32
n_classes=6

learning_rate=0.0025
lambda_loss_amount=0.0015
training_iters=training_data_count*300
batch_size=1500
display_iter=30000


# In[8]:


print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


# In[9]:


#now the utility_function for training
def LSTM_RNN(_X,_weights,_biases):
    #input shape  (batch_size.n_step,n_input)
    _X=tf.transpose(_X,[1,0,2])
    _X=tf.reshape(_X,[-1,n_input])
    #new shape (n_step*batch_size,n_inputs)
    _X=tf.nn.relu(tf.matmul(_X,_weights['hidden'])+_biases['hidden'])
    _X=tf.split(_X,n_step,0)
    lstm_cell_1=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True)
    lstm_cell_2=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True)
    lstm_cells=tf.contrib.rnn.MultiRNNCell([lstm_cell_1,lstm_cell_2],state_is_tuple=True)
    
    outputs,states=tf.contrib.rnn.static_rnn(lstm_cells,_X,dtype=tf.float32)
    lstm_last_output=outputs[-1]
    return tf.matmul(lstm_last_output,_weights['out'])+_biases['out']

#extract the batch size
def extract_batch_size(_train,step,batch_size):
    shape=list(_train.shape)
    shape[0]=batch_size
    batch_s=np.empty(shape)
    for i in range(batch_size):
        index=((step-1)*batch_size+i)%len(_train)
        batch_s[i]=_train[index]
    return batch_s

def one_hot(y_):
    y_=y_.reshape(len(y_))
    n_value=int(np.max(y_))+1
    return np.eye(n_value)[np.array(y_,dtype=np.int32)]


# In[10]:


#now lets builds neural network
x=tf.placeholder(tf.float32,[None,n_step,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])

weights={
    'hidden': tf.Variable(tf.random_normal([n_input,n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases={
    'hidden':tf.Variable(tf.random_normal([n_hidden])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
pred=LSTM_RNN(x,weights,biases)

l2=lambda_loss_amount*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
#l2 loss prevent this neural network to overfit the data
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))+l2
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


# In[13]:


#now train the neural network
test_losses=[]
test_accuracies=[]
train_losses=[]
train_accuracies=[]
sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init=tf.global_variables_initializer()
sess.run(init)

step=1
while step*batch_size<=training_iters:
    batch_xs=extract_batch_size(X_train,step,batch_size)
    batch_ys=one_hot(extract_batch_size(y_train,step,batch_size))
    
    #fit training using batch data
    _,loss,acc=sess.run([optimizer,cost,accuracy],feed_dict={x:batch_xs,y:batch_ys})
    
    #evaluate training only at some stage of data
    if (step*batch_size%display_iter==0) or (step==1) or (step*batch_size>training_iters):
        print("training iter #" + str(step*batch_size) + ": Batch Loss = " + "{:.6f}".format(loss) + ",Accuracy={}".format(acc))
        loss,acc=sess.run([cost,accuracy],feed_dict={x:X_test,y:one_hot(y_test)})
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("performance on test set:"+"Batch loss ={}".format(loss)+"accuracy={}".format(acc))
    step+=1
print("optimization finished")

#accuracy on test dataset
one_hot_predictions,accuracy,final_loss=sess.run([pred,accuracy,cost],feed_dict={x:X_test,y:one_hot(y_test)})
test_losses.append(final_loss)
test_accuracies.append(accuracy)
print("final result "+"batch loss={}".format(final_loss)+"accuracy={}".format(accuracy))


# In[16]:


import matplotlib
get_ipython().magic('matplotlib inline')
font={
    'family':'BitStream Vera sans',
    'weight':'bold',
    'size':18
}

matplotlib.rc('font',**font)
width=12
height=12
plt.figure(figsize=(width,height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()


# In[17]:


#multiclass classification matrix accuracy and confusion matrix
predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[18]:


sess.close()


# In[19]:


# Let's convert this notebook to a README for the GitHub project's title page:
get_ipython().system('jupyter nbconvert --to markdown LSTM.ipynb')
get_ipython().system('mv LSTM.md README.md')

