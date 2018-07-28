import random
from math import log, ceil
from DecisionTree import DecisionTree

class RandomForest(object):
    """
    Class of the Random Forest
    """
    def __init__(self, tree_num):
        self.tree_num = tree_num
        self.forest = []

    def train(self, records, attributes):
        """
        This function will train the random forest, the basic idea of training a
        Random Forest is as follows:
        1. Draw n bootstrap samples using bootstrap() function
        2. For each of the bootstrap samples, grow a tree with a subset of
            original attributes, which is of size m (m << # of total attributes)
        """

        for count in range(0, int(self.tree_num)):
            # Step 1 :Finding out the samples using bootstrap() for every treenum
            sample_rec = self.bootstrap(records)

            # Step 2 : For every treenum selecting 50% of the sample attributes 
            #           at random (without replacement) to be used for tree contruction
            sample_attr = []
            while len(sample_attr) < ceil(0.5 * len(attributes)):
                rand = random.choice(attributes)
                if not rand in sample_attr:
                    sample_attr.append(rand)

            # Creating a new Tree instance, training it based on the records and 
            # attributes bootstraped above and adding to the forest
            Tree = DecisionTree()
            Tree.train(sample_rec, sample_attr)
            self.forest.append(Tree)


    def predict(self, sample):
        """
        The predict function predicts the label for new data by aggregating the
        predictions of each tree.

        This function should return the predicted label
        """
        pois = 0 # number of poisnous predictions
        edible = 0 # number of edible predictions

        # For every tree in the forest, get the label predictions
        for tree in self.forest:
            predicted = tree.predict(sample)
            if predicted == 'p':
                pois += 1
            else:
                edible += 1

        # highest count gives the final predicted label
        if pois >= edible:
            return 'p'
        else:
            return 'e'


    def bootstrap(self, records):
        """
        This function bootstrap will return a set of records, which has the same
        size with the original records but with replacement.
        """
        sample_records=[]
        # add random value until we get same sample length as records
        # we are sampling with replacements
        while len(sample_records) < len(records):
            rand = random.choice(records)
            sample_records.append(rand)

        return sample_records


