import numpy as np
import copy
import argparse 

def parse_args():

    parser = argparse.ArgumentParser(description='Compute logical complexity')
    parser.add_argument('-cat', help='<category/path/filename>', required=True)
    parser.add_argument('-n', help='number of dimensions', required=False)
    args = parser.parse_args()
    return args.cat, args.n

def derivative(category, var):
    """""
    Input(s)
        category: an array containing all binary members of a category
        var: Boolean variable with respect to which the derivative will be computed
    Output(s)
        perturbed_cat: returns the perturbed category resulting from computing the derivative as an array

    This function computes the Boolean partial derivative but does not return the expression representing
    that derivative. Rather, it returns the perturbed category that results from computing a Boolean partial
    derivative. This is computed by flipping the parity of the variable with respect to which the Boolean
    partial derivative is taken in each member of the category
    """
    perturbed_cat = []
    for obj in category:
        perturbed_obj = copy.deepcopy(obj)
        perturbed_obj[var] = int(not perturbed_obj[var])
        perturbed_cat.append(perturbed_obj)
    return perturbed_cat

def logical_manifold(category, derivatives, fraction=False):
    """
    Input(s)
        category: an array containing all binary members of a category
        derivatives: an array of all the perturbed categories obtained from taking the Boolean partial derivatives
                    with respect to each variable
        fraction: this flag allows the user to decide whether or not to return the quotient of count and cardinality
                  It may be useful when inspecting the logical manifold to instead have tuples representing this
                  quotient.
    Output(s)
        logcial_manifold: an array containing the logical norms with respect to each variable

    This function computes the logical manifold of a Boolean category as defined by Vigo (2009)
    """
    logi_manifold = []
    cardinality = len(category)
    for partial in derivatives:
        count = len([obj for obj in partial if obj in category])
        if fraction:
            logi_manifold.append((count, cardinality))
        else:
            logi_manifold.append(count / cardinality)
    return logi_manifold

def logical_complexity(category, n):
    """
    This function computes the categorical complexity by using Boolean categorical invariance theory as well
    as the cardinality of the set representing the category as defined by Vigo (2009)

    Input(s)
        category: an array containing all binary members of a category
    Output(s)
        Returns a tuple containing the complexity of the category, the size of the category and the Euclidian distance
        between the origin logical manifold and the logical manifold of the category
    """
    logi_manifold = logical_manifold(category, [derivative(category, i) for i in range(n)])
    dist = np.linalg.norm(logi_manifold)
    complexity = len(category) / (dist + 1)
    return complexity

def main(**kwargs):

    # Get arguments
    if len(kwargs) == 0:
        cat_fname, n = parse_args()
    else:
        #TODO
        # verify args
        # load categories
        pass

    # Load categories
    categories = np.load(cat_fname)
    catA = categories['a']
    catB = categories['b']

    # Verify n
    if n is None:
        nA = len(catA) 
        nB = len(catB)
    else:
        nA = int(n)
        nB = int(n)

    # Compute logical complexity
    complexityA = logical_complexity(catA, nA)
    complexityB = logical_complexity(catB, nB)
    
    print(f'Category A logical complexity: {complexityA}\nCategory B logical complexity: {complexityB}')

    return complexityA, complexityB 

if __name__ == '__main__':
    main()
