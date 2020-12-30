import nn
import numpy as np
import math
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        return nn.DotProduct(x, self.get_weights())
        "*** YOUR CODE HERE ***"

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        prediction = nn.as_scalar(self.run(x))
        return 1 if prediction >= 0 else -1
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        batch_size = 1
        max_iteration = 1000

        for _ in range(max_iteration):
            temp = 0
            for x, y in dataset.iterate_once(batch_size):
                guess = self.get_prediction(x)
                correct = nn.as_scalar(y)
                if guess != correct:
                    self.get_weights().update(x, correct)
                    temp = 1
            if temp == 0: break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    '''
        Input            Hidden     output        || Layers 
            O      ---      0    -
            O      ---      0       \
            .               .         - 0
            .               .       /
            0      ---      0    -   

            This is Neural Network architecture I'll be using with Hidden Layer having
            activation function (ReLu)
    '''


    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #  ** params **
        self.layer_size =  50
        self.learning_rate = 0.02
        self.acceptable_loss = 0.01
        # default
        self.batch_size = 1

        self.m1 = nn.Parameter(1, self.layer_size)
        self.b1 = nn.Parameter(1, self.layer_size)
        self.m2 = nn.Parameter(self.layer_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def choose_batch_size(self, len_dataset):
        for i in range(len_dataset//10 + 1, 0, -1):
            if len_dataset % i == 0:
                return i

    def input_layer(self, x):
        xm1 = nn.Linear(x, self.m1)
        return nn.AddBias(xm1, self.b1)

    def hidden_layer(self, a):
        ### apply ReLu activation on a
        o = nn.Linear(nn.ReLU(a), self.m2)
        return nn.AddBias(o, self.b2)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        a = self.input_layer(x)
        predicted_y = self.hidden_layer(a)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        self.batch_size = self.choose_batch_size(dataset.y.shape[0])

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_m1, grad_b1, grad_m2, grad_b2 = nn.gradients(loss, [self.m1, self.b1, self.m2, self.b2])
                self.m1.update(grad_m1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.m2.update(grad_m2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)

            if nn.as_scalar(loss) < self.acceptable_loss: break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    '''
                       activation   activation
            
            Input        Hidden-1   Hidden-2    Hidden-3    output        || Layers 
                O   ---     0                             
                O   ---     0          0                    
                .           .          .           0       \ 
                .           .          .           .         - 0
                .           .          .           0       /
                0   ---     0          0
                0   ---     0                              

                This is Neural Network architecture I'll be using with 3 Hidden Layer, 1-2 having
                activation function (ReLu)
    '''
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layer_size = 100
        self.learning_rate = 0.08
        self.acceptable_accuracy = 0.98
        # default
        self.batch_size = 1

        self.m1 = nn.Parameter(784, self.layer_size * 3)
        self.b1 = nn.Parameter(1, self.layer_size * 3)
        self.m2 = nn.Parameter(self.layer_size * 3, self.layer_size * 2)
        self.b2 = nn.Parameter(1, self.layer_size * 2)
        self.m3 = nn.Parameter(self.layer_size * 2, self.layer_size)
        self.b3 = nn.Parameter(1, self.layer_size)
        self.m4 = nn.Parameter(self.layer_size, 10)
        self.b4 = nn.Parameter(1, 10)

    def choose_batch_size(self, len_dataset):
        for i in range(math.ceil(math.sqrt(len_dataset)) // 5 + 1, 0, -1):
            if len_dataset % i == 0:
                return i

    def input_layer(self, x):
        xm1 = nn.Linear(x, self.m1)
        return nn.AddBias(xm1, self.b1)

    def hidden_layer_1(self, a):
        ### apply ReLu activation on a
        o = nn.Linear(nn.ReLU(a), self.m2)
        return nn.AddBias(o, self.b2)

    def hidden_layer_2(self, a):
        ### apply ReLu activation on a
        o = nn.Linear(nn.ReLU(a), self.m3)
        return nn.AddBias(o, self.b3)

    def hidden_layer_3(self, a):
        o = nn.Linear(a, self.m4)
        return nn.AddBias(o, self.b4)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        a = self.input_layer(x)
        a2 = self.hidden_layer_1(a)
        a3 = self.hidden_layer_2(a2)
        predicted_y = self.hidden_layer_3(a3)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        self.batch_size = self.choose_batch_size(dataset.y.shape[0])
        #
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_m1, grad_b1, grad_m2, grad_b2 , grad_m3, grad_b3, grad_m4, grad_b4 = nn.gradients(loss, [self.m1, self.b1, self.m2, self.b2,self.m3, self.b3, self.m4, self.b4])
                self.m1.update(grad_m1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.m2.update(grad_m2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)
                self.m3.update(grad_m3, -self.learning_rate)
                self.b3.update(grad_b3, -self.learning_rate)
                self.m4.update(grad_m4, -self.learning_rate)
                self.b4.update(grad_b4, -self.learning_rate)
            if dataset.get_validation_accuracy() >= self.acceptable_accuracy: return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layer_size = 300
        self.learning_rate = 0.05
        self.acceptable_accuracy = 0.86
        # default
        self.batch_size = 1

        self.m1 = nn.Parameter(self.num_chars, self.layer_size)
        self.b1 = nn.Parameter(1, self.layer_size)

        self.m_x = nn.Parameter(self.num_chars, self.layer_size)
        self.m_h = nn.Parameter(self.layer_size, self.layer_size)
        self.b_x_h = nn.Parameter(1, self.layer_size)

        self.m_x_1 = nn.Parameter(self.num_chars, self.layer_size)
        self.m_h_1 = nn.Parameter(self.layer_size, self.layer_size)
        self.b_x_h_1 = nn.Parameter(1, self.layer_size)

        self.m_o = nn.Parameter(self.layer_size, 5)
        self.b_o= nn.Parameter(1, 5)

    def f_initial(self, x0):
        return nn.ReLU(nn.AddBias(nn.Linear(x0, self.m1), self.b1))

    def f(self, x, h):
        z = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x, self.m_x), nn.Linear(h, self.m_h)), self.b_x_h))
        return  nn.AddBias(nn.Add(nn.Linear(x, self.m_x_1), nn.Linear(z, self.m_h_1)), self.b_x_h_1)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = self.f_initial(xs[0])

        for i in range (1, len(xs)):
            ch = xs[i]
            h = self.f(ch, h)

        output = nn.AddBias(nn.Linear(h, self.m_o), self.b_o)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        return nn.SoftmaxLoss(predicted_y, y)

    def choose_batch_size(self, len_dataset):
        for i in range(math.ceil(math.sqrt(len_dataset)) // 5 + 1, 0, -1):
            if len_dataset % i == 0:
                return i

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        self.batch_size = self.choose_batch_size(dataset.train_y.shape[0])

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_m1, grad_b1, grad_m_x, grad_m_h, grad_b_x_h,grad_m_x_1, grad_m_h_1, grad_b_x_h_1, grad_m_o, grad_b_o = nn.gradients(loss, [self.m1, self.b1, self.m_x, self.m_h, self.b_x_h,self.m_x_1, self.m_h_1, self.b_x_h_1, self.m_o, self.b_o])
                self.m1.update(grad_m1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.m_x.update(grad_m_x, -self.learning_rate)
                self.m_h.update(grad_m_h, -self.learning_rate)
                self.b_x_h.update(grad_b_x_h, -self.learning_rate)
                self.m_x_1.update(grad_m_x_1, -self.learning_rate)
                self.m_h_1.update(grad_m_h_1, -self.learning_rate)
                self.b_x_h_1.update(grad_b_x_h_1, -self.learning_rate)
                self.m_o.update(grad_m_o, -self.learning_rate)
                self.b_o.update(grad_b_o, -self.learning_rate)
            if dataset.get_validation_accuracy() >= self.acceptable_accuracy: return
