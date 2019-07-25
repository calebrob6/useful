def assert_cnf_shape(cnf):
    assert len(cnf.shape) == 2
    assert cnf.shape[0] == cnf.shape[1]
    assert np.all(cnf>=0)
    
def confusion_matrix(y_true, y_pred, class_values):
    ''' y_true and y_pred should either be equally sized 1D or 2D arrays where entries indicate class values
    
    cnf[i,j] will be the number of entries that are labeled as class_values[i] in y_true, and labeled as class_values[j] in y_pred
    '''
    if isinstance(y_true, list):
        assert isinstance(y_pred, list)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert len(y_true.shape) == 1 and len(y_pred.shape) == 1
    
    assert len(y_true.shape) == len(y_pred.shape)
    assert len(y_true.shape) <= 2 #
    for i in range(len(y_true.shape)):
        assert y_true.shape[i] == y_pred.shape[i]
    
    cnf = np.zeros((len(class_values), len(class_values)), dtype=np.int)
    
    for i in range(len(class_values)):
        class_i = class_values[i]
        mask = (y_true==class_i)
        for j in range(len(class_values)):
            class_j = class_values[j]
            cnf[i,j] = np.sum(mask & (y_pred==class_j))
    return cnf    

def mean_per_class_accuracy(cnf):
    assert_cnf_shape(cnf)
    
    accuracies = []
    for i in range(cnf.shape[0]):
        row_sum = np.sum(cnf[i,:])
        if row_sum != 0:
            accuracies.append(cnf[i,i] / row_sum)
    return np.mean(accuracies)

def miou(cnf):
    assert_cnf_shape(cnf)
    
    ious = []
    for i in range(cnf.shape[0]):
        iou = cnf[i,i] / (np.sum(cnf[i,:] + np.sum(cnf[:,i]) - cnf[i,i]))
        ious.append(iou)
    return np.mean(ious)

def accuracy(cnf):
    assert_cnf_shape(cnf)
    
    num_correct = 0
    for i in range(cnf.shape[0]):
        num_correct += cnf[i,i]
    
    return num_correct / np.sum(cnf)
