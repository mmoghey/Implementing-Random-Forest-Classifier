import random
from math import log, ceil

class TreeNode(object):
    def __init__(self, isLeaf=False):
        self.isLeaf = isLeaf
        self.label = None # stores the label information for leaf node else None
        self.attribute = None # stores the best attribute chosen during training for non-leaf node else None
        self.attr_val = None # stores the best split value chosen during training for non-leaf node else None
        self.children = [] # stores the child nodes information for non-leaf node else None

    def predict(self, sample):
        """
        This function predicts the label for a particular sample. 
        If this is the leaf node then returns its label
        If this is not a leaf then it travers down the decision tree
            to find out the predicted value
        """
        predicted = None
        if self.isLeaf == True:
            predicted = self.label

        else:
            if self.attr_val == sample["attributes"][self.attribute]:
                predicted = self.children[0].predict(sample)
            else:
                predicted = self.children[1].predict(sample)

        return predicted

    def printNode(self, counter):
        """
        This function prints the information of a particular node of the tree
        """
        space = "    "
        for i in range(0,counter):
            space += "    "

        if self.isLeaf == False:
            print '%s %s %s ' %(space, self.attribute, self.attr_val)
        else:
            print '%s label : %s' %(space,self.label)




class DecisionTree(object):
    """
    Class of the Decision Tree
    """
    def __init__(self):
        self.root = None

    def train(self, records, attributes):
        """
        This function trains the model with training records "records" and
        attribute set "attributes", the format of the data is as follows:
            records: training records, each record contains following fields:
                label - the lable of this record
                attributes - a list of attribute values
            attributes: a list of attribute indices that you can use for
                        building the tree
        Typical data will look like:
            records: [
                        {
                            "label":"p",
                            "attributes":['p','x','y',...]
                        },
                        {
                            "label":"e",
                            "attributes":['b','y','y',...]
                        },
                        ...]
            attributes: [0, 2, 5, 7,...]
        """
        self.root = self.tree_growth(records,attributes)
      
        #If we want to print tree to look at the created tree
        #print "Records Length: %s Attribute len: %s" %(len(records), len(attributes))
        #print("*******Printing Decision Tree*******")
        #self.printTree(self.root, 0)

    def printTree(self, Node, counter):
        """
        This function prints the tree by calling print on every node of the tree
        """

        Node.printNode(counter)

        if Node.isLeaf == False:
            if Node.children[0] != None:
                self.printTree(Node.children[0], counter + 1)
            if Node.children[1] != None:
                self.printTree(Node.children[1], counter + 1)



    def predict(self, sample):
        """
        This function predict the label for new sample by calling the predict
        function of the root node
        """
        return self.root.predict(sample)

    def stopping_cond(self, records, attributes):
        """
        The stopping_cond() function is used to terminate the tree-growing
        process by testing whether all the records have either the same class
        label or the same attribute values.

        This function should return True/False to indicate whether the stopping
        criterion is met
        """
        edible = 0 #track count of edible mushrooms 
        pois = 0 #tracks count of poisnous mushrooms

        # get the counts
        for sample in records:
            if sample["label"] == 'e' :
                edible += 1
            else :
                pois += 1
        # If all class labels are e or all class labels are p then return True 
        # else if there are no attributes left to split then return True  
        # else if all the attribute values match but label values are different then return True 
        # else stopping condition is not met return False
        if edible == len(records) :
            return True
        elif pois == len(records) :
            return True
        elif len(attributes) == 0:
            if edible >= pois:
                return True
            else:
                return True
        else :
            i = 0
            j = 0
            match = True
            for sample in records[1:]:
                if sample["attributes"][attributes[j]] != records[0]["attributes"][attributes[j]]:
                    match = False
                    break

            if match == True:
                if edible >= pois:
                    return True
                else:
                    return True

            return False



    def classify(self, records):
        """
        This function determines the class label to be assigned to a leaf node.
        In most cases, the leaf node is assigned to the class that has the
        majority number of training records

        This function should return a label that is assigned to the node
        """
        edible = 0 # count edible label
        pois = 0 # count poisnous label

        for sample in records:
            if sample["label"] == 'e' :
                edible += 1
            else :
                pois += 1

        # return the label with majority of training records
        if edible >= pois:
            return 'e'
        else:
            return 'p'


    def entropy(self, records):
        """
        The entropy() function determines the impurity measure to be able 
        to decide which attribute should be used for splitting and hence
        determining the goodness of split

        This function returns the entropy value of the set of records at a node
        """
        if len(records) == 0:
            return 0

        edible = [e for e in records if e["label"] == 'e'] #fill the edible list
        pois = [p for p in records if p["label"] == 'p'] # fill the poisnous list

        if len(edible) == 0 or len(pois) == 0:
            return 0

        #calculate probabilities
        prob_e = float(len(edible))/float(len(records))
        prob_p = float(len(pois))/float(len(records))

        #calculate and return entropy
        entropy_val = (-1) * prob_e * log(prob_e, 2.0) + (-1) * prob_p * log(prob_p, 2.0)
        return entropy_val


    def gain(self, records, attribute, value):
        """
        The gain() function determines the Information Gain we get when we split the attributes
        This function returns the gain achieved
        """
        if len(records) == 0:
            return 0.0

        recordsL=[]
        recordsR=[]

        for row in records:
            if row["attributes"][attribute] == value:
                recordsL.append(row)
            else:
                recordsR.append(row)

        probL = float(len(recordsL))/float(len(records))
        probR = float(len(recordsR))/float(len(records))

        entropyL = float(self.entropy(recordsL))
        entropyR = float(self.entropy(recordsR))
        entropyP = float(self.entropy(records))

        gain_val = entropyP - (float(entropyL * probL) + float(entropyR * probR))

        return gain_val


    def find_best_split(self, records, attributes):
        """
        The find_best_split() function determines which attribute should be
        selected as the test condition for splitting the trainig records.

        This function should return multiple information:
            attribute selected for splitting
            threshhold value for splitting
            left subset
            right subset
        """

        value = None
        best_attr = (None, None, 0.0) #Stores the best attribute, its split value and gain
        for attr in attributes:
            #Find best value for split among all values of an attribute
            val_list = [row["attributes"][attr] for row in records if 1]
            attr_values = list(set(val_list))
            best_val = (None, 0.0) #stores the best value and its gain

            val_gain = 0.0
            for val in attr_values:
                val_gain = self.gain(records, attr, val)
                if val_gain >= best_val[1]:
                    best_val = (val, val_gain)
            #This best value for split will give the max gain for this attribute
            value = best_val[0]

            #calculate gain for the attribute with the above best split value
            attr_gain = self.gain (records, attr, value)

            if attr_gain >= best_attr[2]:
                best_attr = (attr, value, attr_gain)

        left_subset = []
        right_subset = []
        for row in records:
            if row["attributes"][best_attr[0]] == best_attr[1]:
                left_subset.append(row)
            else:
                right_subset.append(row)
        return best_attr[0], best_attr[1], left_subset, right_subset


    def tree_growth(self, records, attributes):
        """
        This function grows the Decision Tree recursively until the stopping
        criterion is met. Please see textbook p164 for more details

        This function should return a TreeNode
        """
        # Your code here
        # Hint-1: Test whether the stopping criterion has been met by calling function stopping_cond()
        # Hint-2: If the stopping criterion is met, you may need to create a leaf node
        # Hint-3: If the stopping criterion is not met, you may need to create a
        #         TreeNode, then split the records into two parts and build a
        #         child node for each part of the subset
        root = TreeNode()
        sample_records=[]

        # Sample 75% records at random with replacement (this will be done for each node in recursion)
        while len(sample_records) < ceil( 0.75 * len(records)):
            rand = random.choice(records)
            sample_records.append(rand)

        stop = self.stopping_cond(sample_records,attributes)

        # If stopping condition is met then add a leaf node with predicted label
        if stop == True:
            root.label = self.classify(records)
            root.isLeaf = True
        else:
            # Stopping condition not met then grow the tree
            attr, value, left_subset, right_subset = self.find_best_split(sample_records, attributes)

            root.attribute = attr
            root.attr_val = value

            # remove the attribute chosen from the list, new_attr will be sent to child nodes
            new_attr = attributes[:]
            new_attr.remove(attr)

            childL = self.tree_growth(left_subset, new_attr)
            childR = self.tree_growth(right_subset, new_attr)

            #Add both childs to the parent node
            root.children.append(childL)
            root.children.append(childR)

        return root

