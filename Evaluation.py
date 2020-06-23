class Evaluations():
    def __init__ (self,
                   history,
                   y_pred,
                   y_true,
                   loss_score,
                   error,
                   validation_score):
        self.history = history
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss_score = loss_score
        self.error = error
        self.validation_score = validation_score
        
        self.matrix,self.kappa = self.matrix_and_kappa(self.y_pred,self.y_true)
        self.f1 = self.f1_score(self.y_pred,self.y_true)
        
    # Matrix confusion and the kappa value
    def matrix_and_kappa(self,y_pred,y_true):
        for i in y_pred:
            if i[0]>i[1]:
                i[0]=1
                i[1]=0
            else:
                i[1]=1
                i[0]=0
        y_pred = np.array(y_pred)
        y_pred_1 = []
        y_true_1 = []
        for i in y_pred:
            if i[0]==1:
                y_pred_1.append(1)
            else:
                y_pred_1.append(2)
        for i in y_true:
            if i[0]==1:
                y_true_1.append(1)
            else:
                y_true_1.append(2)
        C2 = confusion_matrix(y_true_1,y_pred_1)
        kappa_value = cohen_kappa_score(y_true_1, y_pred_1)
        return C2,kappa_value
    
    # f1
    def f1_score(self,y_pred,y_true):
        for i in y_pred:
            if i[0]>i[1]:
                i[0]=1
                i[1]=0
            else:
                i[1]=1
                i[0]=0
        y_pred = np.array(y_pred)
        y_pred_1 = []
        y_true_1 = []
        for i in y_pred:
            if i[0]==1:
                y_pred_1.append(1)
            else:
                y_pred_1.append(2)
        for i in y_true:
            if i[0]==1:
                y_true_1.append(1)
            else:
                y_true_1.append(2)
        f1 = f1_score(y_true,y_pred, average = None)
        return f1
    
    def draw_pict(self,history,types = 1):
        if types == 1:
            plt.plot(history.history['binary_accuracy'])
            plt.plot(history.history['val_binary_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='lower right')  
            plt.title('accuracy')
            plt.show()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss') 
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='upper right')  
            plt.title('loss')
            plt.show()
        else:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='lower right')  
            plt.title('accuracy')
            plt.show()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss') 
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='upper right')  
            plt.title('loss')
            plt.show()
        return
