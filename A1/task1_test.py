# No external libraries are allowed to be imported in this file
import random
import numpy as np

def mock_datastream():
    """This function is a mock datastream generator. It yields transactions one by one.
    It is used for testing the reservoir_sampling function. It is not allowed to change
    this function.

    Yields:
        transactions: A transaction from the datastream
    """
    mu = 10
    for t in range(10_000):
        mu = 10 + 0.001 * t   # drift progressivo della media
        yield random.gauss(mu, 100)

def reservoir_sampling(k, datastream):
    """This function should contain the code for the reservoir sampling algorithm.
    As an input it takes the sample size k and a datastream which is a generator 
    that yields the transactions one by one. Note that the resulting sample should be
    representative of the whole datastream.

    Args:
        k (int): The sample size
        datastream (func): The datastream generator that yields the transactions one by one

    Returns:
        list[transactions]: A list of size k containing the sampled transactions.
    """
    sample = []
    for index, transaction in enumerate(datastream()):
        # transaction, contains the current transaction from the stream
        # Note that it is NOT allowed to store the whole datastream in memory
        # Note that the sample array size should not exceed k

        # BEGIN IMPLEMENTATION

        elif x<p:
            sample[isaac]=transaction
        
    print(f"\n{np.mean(sample)}")
        # END IMPLEMENTATION

    return sample, np.mean(sample)

if __name__ == "__main__":
    # You can use this main section for testing the reservoir_sampling function
    test=[]
    for _ in range (100):
        sample, mean = reservoir_sampling(5000, mock_datastream)
        test.append(mean)
    #print(sample)
    print(f"In 100 sample, the average mean is: \n{np.mean(mean)}")