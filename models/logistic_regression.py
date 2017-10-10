import math
import numpy as np

def read_dataset(path, name='logistic', data_only = False):

    """
    Given a .csv file path, separates the responsevector y from the data matrix X.
    Assumes all columns have headings, are numerical, and the response vector is the first column.
    Returns an instance of LogisticModel with the following attributes:
        name: provided name for model
        varnames: tuple of variable names
        y: the response vector,
        X: the predictor matrix, with an additional dummy variable of 1's for the intercept coefficient

    Alternatively, returns the numpy arrays only as a tuple (y,X)
    """

    data_struc = np.genfromtxt(path, dtype=float, names=True, delimiter=",")
    data_array = data_struc.view((float, len(data_struc.dtype.names)))
    split_data = np.split(data_array, indices_or_sections=[1], axis=1)
    varnames = data_struc.dtype.names
    y = split_data[0].flatten()
    X = np.concatenate((np.ones_like(split_data[0].flatten(), dtype='float64')[:, np.newaxis], split_data[1]), axis=1)

    if data_only:
        return (y,X)
    else:
        return(LogisticModel(name,varnames,y,X))


class LogisticModel(object):
    """A logistic regression model for fitting and predicting binary response data.

    Attributes:
        name: provided name of model
        varnames: tuple of variable names
        y: the response vector,
        X: the predictor matrix, with an additional dummy variable of 1's for the intercept coefficient
    """

    def __init__(self, name, varnames, y, X):
        self.name = name
        self.varnames = varnames
        self.y = y
        self.X = X

    def fit(self, iterations=25):
        """
        Given a response vector (y), training data matrix (X), runs the IRLS algorithm to the specified number of iterations.
        Returns a dictionary containing the coefficients
        """

        w = np.array([0]*self.X.shape[1], dtype='float64')
        y_bar = np.mean(self.y)
        w_init = math.log(y_bar/(1-y_bar))
        self.converged = False
        nll_sequence = []
        for i in range(iterations):
            h = self.X.dot(w)
            p = 1/(1+np.exp(-h))
            p_adj = p
            p_adj[p_adj==1.0] = 0.99999999
            nll = -(1-self.y.dot(np.log(1-p_adj)))+self.y.dot(np.log(p_adj))
            nll_sequence += [nll]

            if i>1:
                if not self.converged and abs(nll_sequence[-1]-nll_sequence[-2])<.000001:
                    self.converged = True
                    self.converged_k = i+1

            s = p*(1-p)
            S = np.diag(s)
            arb_small = np.ones_like(s, dtype='float64')*.000001
            z = h + np.divide((self.y-p), s, out=arb_small, where=s!=0)
            Xt = np.transpose(self.X)
            XtS = Xt.dot(S)
            XtSX = XtS.dot(self.X)
            inverse_of_XtSX = np.linalg.inv(XtSX)
            inverse_of_XtSX_Xt = inverse_of_XtSX.dot(Xt)
            inverse_of_XtSX_XtS = inverse_of_XtSX_Xt.dot(S)
            w = inverse_of_XtSX_XtS.dot(z)

        self.nll = nll
        self.nll_sequence = nll_sequence
        self.w=w

        if not self.converged:
            print('Warning: IRLS failed to converge. Try increasing the number of iterations.')

        return(self)

    def summary(self):
        """
        Prints a formatted table of the model coefficients
        """

        if not hasattr(self, 'w'):
            print('LogisticModel has not been fit.')
            return(None)

        coef_labels = ['---------------','<Intercept>']+list(self.varnames[1:])
        estimates = ['---------------']+list(self.w)

        # This table will eventually contain more metrics
        table_dic = dict(zip(coef_labels, estimates))

        coef_str = ' + '.join(self.varnames[1:])+'\n'

        print('\n'+self.name+': logistic regression')
        print('\n{} ~ {}'.format(self.varnames[0], coef_str))
        print('\033[1m'+"{:<15} {:<15}".format('Coefficient','Estimate')+'\033[0m')
        for k, v in sorted(table_dic.items()):
            label = v
            print("{:<15} {:<15}".format(k, label))
        if not self.converged:
            print('\nWarning: IRLS failed to converge. Try increasing the number of iterations.')
        else:
            print('\nConverged in {} iterations (IRLS)'.format(self.converged_k))

        return(None)


    def predict(self, X, use_probability = False):
        """
        Given the fitted model and a new sample matrix, X,
        returns an array (y) of predicted log-odds (or optionally the probabilities).
        """

        if not hasattr(self, 'w'):
            print('LogisticModel has not been fit.')
            return(None)

        pred = X.dot(self.w)

        if use_probability:
            odds = np.exp(pred)
            pred = odds / (1 + odds)

        return(pred)
