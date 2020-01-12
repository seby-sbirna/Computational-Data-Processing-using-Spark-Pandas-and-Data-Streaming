## Project 2 - Web Traffic Analysis upon continuous streams of data

### _by Sebastian Sbirna, Yingrui Li and Aijie Shu_
---

**This is the second of three mandatory projects to be handed in as part of the assessment for the course 02807 Computational Tools for Data Science at Technical University of Denmark, autumn 2019.**

## Introduction
In this project your task is to analyze a stream of log entries. A log entry consists of an [IP address](https://en.wikipedia.org/wiki/IP_address) and a [domain name](https://en.wikipedia.org/wiki/Domain_name). For example, a log line may look as follows:

`192.168.0.1 somedomain.dk`

One log line is the result of the event that the domain name was visited by someone having the corresponding IP address. Your task is to analyze the traffic on a number of domains. Counting the number of unique IPs seen on a domain doesn't correspond to the exact number of unique visitors, but it is a good estimate.

Specifically, you should answer the following questions from the stream of log entries.

- How many unique IPs are there in the stream?
- How many unique IPs are there for each domain?
- How many times was IP X seen on domain Y? (for some X and Y provided at run time)

**The answers to these questions can be approximate!**

You should also try to answer one or more of the following, more advanced, questions. The answers to these should also be approximate.

- How many unique IPs are there for the domains $d_1, d_2, \ldots$?
- How many times was IP X seen on domains $d_1, d_2, \ldots$?
- What are the X most frequent IPs in the stream?

You should use algorithms and data structures that you've learned about in the lectures, and you should provide your own implementations of these.

Furthermore, you are expected to:

- Document the accuracy of your answers when using algorithms that give approximate answers
- Argue why you are using certain parameters for your data structures

This notebook is in three parts. In the first part you are given an example of how to read from the stream (which for the purpose of this project is a remote file). In the second part you should implement the algorithms and data structures that you intend to use, and in the last part you should use these for analyzing the stream.

## Reading the stream
The following code reads a remote file line by line. It is wrapped in a generator to make it easier to extend. You may modify this if you want to, but your solution should remain parametrized, so that your notebook can be run without having to consume the entire file.


```python
import urllib
import mmh3
import math
import pandas as pd

def stream(n):
    i = 0
    with urllib.request.urlopen('https://files.dtu.dk/fss/public/link/public/stream/read/traffic_2?linkToken=_DcyO-U3MjjuNzI-&itemName=traffic_2') as f:
        for line in f:
            element = line.rstrip().decode("utf-8")
            yield element
            i += 1
            if i == n:
                break
```

Here, we write a function that will retrieve a new stream of the specified size (i.e. `STREAM_SIZE`). 

More specifically, it is a generator object will efficiently retrieve the first `STREAM_SIZE` elements from the given URL (opened in the function above).


```python
STREAM_SIZE = 1000000
def get_stream():
    web_traffic_stream = stream(STREAM_SIZE)
    
    return web_traffic_stream
```

---
## Data structures

### Question 1, 2 & 4
**For Questions 1, 2 and 4**, we have identified that the problems could be solved efficiently by using one or more Hyperloglog count computations, which should drastically reduce size and memory requrirements necessary for computing the number of distinct elements within a string.

Since we will need to store the different paramenters of every HLL (hyperloglog) function separate from one another, we have created a simple HLL class, which can be instantiated just as any other object in Python, since we have created its `__init__` function, which will automatically be called when a new object of type HLL is created. The methods for this object will be exactly the ones necessary to update the buckets of the HLL which store the highest number of zeros that was found in any hashed element that passed through it (_add function_), and also to retrieve, starting from this number, the number of unique elements that have passed through the HLL (_count function_), with a mean error of 1.625%, and up to ~4.875% in rare situations.

It needs to be mentioned that our implementation of the HLL algorithm uses only one hashing function, however there are other implements which use a list of multiple hashing functions. <br> Such implementations are slightly more robust to potential random errors of finding a hash with very large number of zeros in the beginning of the stream.

_Error numbers were retrieved from the last thread response from: https://github.com/ascv/HyperLogLog/issues/28_

_The algorithm is highly based on the implementation found in the paper __HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm (Flajolet et al.)__, found here: http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf. Other understandings have been gathered from many articles and resources available online._


```python
class HLL(object):
    
    # Initialization
    def __init__(self):
        self.p = 9  # p is the precision argument, i.e. the number of bits
        self.m = 2 ** self.p  # m is the number of registers in the array M (i.e. the number of buckets)

        if self.p == 4:
            self.alpha = 0.673
        elif self.p == 5:
            self.alpha = 0.697
        elif self.p == 6:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

        self.M = [0 for _ in range(self.m)]  # initialize the array M with m registers (i.e. buckets)
        self.E = 0  # initialize the variable to hold the result

    # Aggregation
    # Add the element to the set represented by this HyperLogLog.
    def add(self, element):
        hashedValue = mmh3.hash(str(element))
        binValue = '{0:032b}'.format(hashedValue).replace("-", "0")  # after

        # divide the binary number into the 2 parts, index and value
        idx = binValue[len(str(binValue)) - self.p:]
        w = binValue[:len(str(binValue)) - self.p]

        leadingzeros = self.countzeros(w)  # count the number of leading 0's
        idx_int = int(idx, 2)  # convert the index to base 10
        self.M[idx_int] = max(self.M[idx_int], leadingzeros)  # compare to number in register and insert the biggest one

    # Count the number of leading 0s in a string
    def countzeros(self, input):
        str_input = str(input)
        cnt = 0
        for index in range(0, len(str_input) - 1):
            if str_input[index] == "0":
                cnt += 1
            else:
                break
        return cnt

    # Computation
    # Should return an estimate of the current number of (distinct) elements in the set.
    def count(self):
        E = self.alpha * float(self.m ** 2) / sum(math.pow(2.0, -x) for x in self.M)

        # V = self.M.count(0)
        V = 0
        for i in range(len(self.M)):
            if self.M[i] == 0:
                V += 1

        if E <= (5.0 / 2.0) * float(self.m):
            if V != 0:
                E = self.linearCounting(self.m, V)
            else:
                E = E
        elif E <= (1.0 / 30) * (2 ** 32):
            E = E
        else:
            E = (-2 ** 32) * math.log(1.0 - (E / (2 ** 32)))
        return int(E)

    def linearCounting(self, m, V):
        return m * math.log(float(m) / float(V))  # the same as m * math.log(m/V)

    # Should return a new hyperloglog that corresponds to the union(merge) of this HLL object and another (i.e. "other").
    def __add__(self, other):
        h = HLL()
        for i in range(self.m):
            h.M[i] = max(self.M[i], other.M[i])
        return h
```

### Question 3 & 5
**For Questions 3 and 5,** if the X, Y are given during compiling time, the frequency of X given Y(s) can be solved by one simple counter. In this question, we assume the X, Y are specified at an arbitray time during the stream is processing. Therefore, the **Count Min Sketch** is used here because it records the counts for any given X.

As streaming data could only be read once, we need to record the amount of times that one IP passes by. Known the name of a specific IP address, we use different hash functions to map it to a table where width is the linear space for all the hashed IP coming in, depth is the number of functions we utilize. The fact we can not ignore is: several IP addresses could be mapped into one place where collisions happen. Thus, using hash functions will **almost always overestimate** the amount of IP. Hence, we need numerous hash functions to get different frequency, picking the minimum one will obviously give us the most precise estimate overall.

### Reason of bad performance of Count Min Sketch

Truly, the top 10 most frequent IPs in the first 1000000 pieces of data in stream are:

['72.187.84.158', '108.41.112.108', '204.141.72.187', '53.30.199.128', '55.29.199.128', '56.29.200.127', '55.31.199.127', '54.30.199.128', '53.32.200.127', '53.29.198.127']
    
Relevant frequency is:

[222, 199, 139, 34, 34, 33, 32, 31, 30, 30]

In our case, <font color='red'>the frequency of IPs doesn't varies so much shown in accuracy test on Question 6 on 1 million IPs. In other words, count mean sketch algorithm is not expected to show a good performance because hash collision affects the low-frequency elements a lot.</font> Therefore, the **Count Mean Min Sketch** algorithm is introduced to improve the accuracy on the long run. In Count Mean Min Sketch algorithm, the noise caused by hash collision is deducted and minimum-based picking is replaced by median-based picking.


```python
from array import array
import random
import numpy as np

class CountMinSketch(object):
    """
    `w`: Width(size) of table.
    `d`: Depth of table, which is the number of hash functions.
    """
    
    def __init__(self, w=9919, d=10):
        self.w = w
        self.d = d
        self.counts = [array('L', (0 for _ in range(self.w))) for _ in range(self.d)]
        self.hash_functions = [lambda x: mmh3.hash(x, seed = k) for k in range(self.d)]
        
    def get_columns(self, a):
        for hash_i in self.hash_functions:
            yield hash_i(a) % self.w
        
    def update(self, a, val=1):
        for row, col in zip(self.counts, self.get_columns(a)):
            row[col] += val
    
    def query(self, a):
        return min(row[col] for row, col in zip(self.counts, self.get_columns(a)))
    
    def __getitem__(self, a):
        return self.query(a)
    
    def __setitem__(self, a, val):
        for row, col in zip(self.counts, self.get_columns(a)):
            row[col] = val
```


```python
from array import array
import random
import numpy as np

class CountMeanMinSketch(object):
    """
    `w`: Width(size) of table.
    `d`: Depth of table, which is the number of hash functions.
    """
    
    def __init__(self, w=9919, d=10):
        self.w = w
        self.d = d
        self.counts = [array('L', (0 for _ in range(self.w))) for _ in range(self.d)]
        self.hash_functions = [lambda x: mmh3.hash(x, seed = k) for k in range(self.d)]
        
    def get_columns(self, a):
        for hash_i in self.hash_functions:
            yield hash_i(a) % self.w
        
    def update(self, a, val=1):
        for row, col in zip(self.counts, self.get_columns(a)):
            row[col] += val
    
    def query(self, a):
        hash_vals_a = []
        for row, col in zip(self.counts, self.get_columns(a)):
            noise = np.mean(row)#(np.sum(row) - row[col]) / (self.w - 1)         # calculate noise on all elements of that row except row[col]
            hash_vals_a.append(row[col] - noise)                    # supercede noise
        return int(round(np.median(hash_vals_a)))                   # get the median

    def __getitem__(self, a):
        return self.query(a)
    
    def __setitem__(self, a, val):
        for row, col in zip(self.counts, self.get_columns(a)):
            row[col] = val
```

### Question 6
**For Question 6**, a Min Heap in which the IP with minimum frequency among the X most frequently IP addresses is maintained. The algorithm is described as below.
1. With the incoming stream, the Count Mean Min Sketch keep updating the 2D array (width x depth).
2. Maintain a min heap with length k
3. Each time a new IP stream in, the sketch increase 1. 
4. - If the new IP has been recorded by the heap, the frequency of that IP in the heap increase by 1, then update the min heap in order to ensure the IP with the minimum frequency among X most frequent IPs is on the root node.
   - If the new IP haven't been encountered before, compare its frequency with the root node of min heap. 
       - If the frequency of new IP is greater than the frequency of root node, replace the root node and update the heap. Otherwise, skip this IP and check next new IP.

**remind that,** for the `https://files.dtu.dk/fss/public/link/public/stream/read/traffic_2?linkToken=_DcyO-U3MjjuNzI-&itemName=traffic_2` data stream, the Count Mean Min Sketch <font color='red'>also do not performs very well</font> because the low frequency of X most frequently IPs. So in the future, Lossy Counting and Space Saving algorithm can be explored.

---
## Analysis

### _Q1: How many unique IPs are there in the stream?_


```python
def Q1_retrieve_unique_ip_count(web_traffic_stream):
    hll = HLL()                                        # Instantiate 1 HLL object, which will estimate the number of unique IPs that we are passing through it

    for entry in web_traffic_stream:
        c_ip = entry.split('\t')[0]                    # Here we store the IP of the entry
        c_domain = entry.split('\t')[1]                # Here we store the web domain of the entry
        hll.add(c_ip)                                  # We update the HLL object by calling its 'add' function, which passes a new entry through it and updates its buckets
        
    return hll.count()
```


```python
approx_unique_ip_count = Q1_retrieve_unique_ip_count(get_stream())

print('Estimated total count of unique IPs in the stream: {}'.format(approx_unique_ip_count))
```

    Estimated total count of unique IPs in the stream: 944333
    

### _Accuracy test: Q1_


```python
## This is a test for computing the accuracy of the algorithm.
## We are using pandas DataFrames for performing quick computations of the exact number requested by the question (Q1) 
## Saving data in memory through DataFrame is not efficient, and this is only meant to be seen as a benchmarking and testing field, which would not exist in a "production" environment

web_traffic_stream = get_stream()

# Create a DataFrame for storing the traffic data, and split the entries into IP column and Domain column by their tab separator
traffic_df = pd.DataFrame(list(s.split('\t') for s in web_traffic_stream), columns = ['ip', 'domain'])

# Compute the exact number of unique IPs in the stream (by using the DataFrame):
true_unique_ip_count = len(traffic_df.ip.unique())

print('Exact count of the unique number of IPs in the stream: {}'.format(true_unique_ip_count))
```

    Exact count of the unique number of IPs in the stream: 971745
    


```python
print('Error difference between HLL algorithm and real-world answer: {}%'
      .format(abs(round(((1 - (approx_unique_ip_count / true_unique_ip_count)) * 100), 3))))
```

    Error difference between HLL algorithm and real-world answer: 2.821%
    

### _Q2: How many unique IPs are there for each domain?_


```python
def Q2_retrieve_unique_ip_per_domain(web_traffic_stream):

    # Since we can expect that the total number of popular domains visited is going to be similar across people, the number of domains encountered is expected to be quite small
    # Therefore, it is computationally reasonable to say that keeping a dictionary with the keys as being the unique domains is computationally efficient
    # The values in this dictionary will be a Hyperloglog object for each unique key (i.e. domain), so that we can count how many unique IPs are found for each individual domain

    domains_dict = {} # Initialize the domain dictionary

    for entry in web_traffic_stream:
        c_ip = entry.split('\t')[0]
        c_domain = entry.split('\t')[1]

        if (not domains_dict.get(c_domain)):                        # If the key does not yet exist in the domain dictionary, 
            domains_dict[c_domain] = HLL()                          # Add the new domain to the dictionary, and set its value to be a new hyperloglog object
        domains_dict[c_domain].add(c_ip)                            # We update the HLL object of a corresponding domain, by calling its 'add' function upon each IP that visited the domain

    for domain, hll in domains_dict.items():                        # For each individual domain, we will print the unique number of items counted by the corresponding HLL object
        print('Domain: {}, approximate count: {}'.format(domain, hll.count()))
        
    return domains_dict                                             # The only purpose why we return the domain dictionary is to be able to perform error benckmarking on it 
```


```python
# In domains domains_dict, we will save the HLL object for each domain, so we can reuse it when computing error benchmarking
domains_dict = Q2_retrieve_unique_ip_per_domain(get_stream())
```

    Domain: python.org, approximate count: 265174
    Domain: wikipedia.org, approximate count: 480675
    Domain: pandas.pydata.org, approximate count: 120872
    Domain: dtu.dk, approximate count: 26610
    Domain: google.com, approximate count: 25719
    Domain: databricks.com, approximate count: 13129
    Domain: github.com, approximate count: 12531
    Domain: spark.apache.org, approximate count: 5104
    Domain: datarobot.com, approximate count: 2762
    Domain: scala-lang.org, approximate count: 1
    

### _Accuracy test: Q2_


```python
for domain in traffic_df.domain.unique():
    approx_unique_ip_nr_by_domain = domains_dict[domain].count()
    true_unique_ip_nr_by_domain = len(traffic_df[traffic_df.domain == domain].ip.unique())
    
    print('Domain: {}, true unique IP count: {}'.format(domain, true_unique_ip_nr_by_domain))
    print('(Error difference: {}%'.format(abs(round(((1 - (approx_unique_ip_nr_by_domain / true_unique_ip_nr_by_domain)) * 100),3))))
```

    Domain: python.org, true unique IP count: 256386
    (Error difference: 3.428%
    Domain: wikipedia.org, true unique IP count: 510191
    (Error difference: 5.785%
    Domain: pandas.pydata.org, true unique IP count: 128723
    (Error difference: 6.099%
    Domain: dtu.dk, true unique IP count: 26144
    (Error difference: 1.782%
    Domain: google.com, true unique IP count: 26082
    (Error difference: 1.392%
    Domain: databricks.com, true unique IP count: 13143
    (Error difference: 0.107%
    Domain: github.com, true unique IP count: 12788
    (Error difference: 2.01%
    Domain: spark.apache.org, true unique IP count: 5217
    (Error difference: 2.166%
    Domain: datarobot.com, true unique IP count: 2559
    (Error difference: 7.933%
    Domain: scala-lang.org, true unique IP count: 1
    (Error difference: 0.0%
    

### _Q3: How many times was IP X seen on domain Y? (for some X and Y provided at run time)_


```python
def Q3_count_IP_X_in_domain_Y(web_traffic_stream, X, Y):
    domains_dict = {}
    
    for entry in web_traffic_stream:
        
        c_ip = entry.split('\t')[0]
        c_domain = entry.split('\t')[1]
        
        if (not domains_dict.get(c_domain)): # If the domain does not yet exist in the domain dictionary, 
            domains_dict[c_domain] = CountMeanMinSketch(w = 49999, d = 10)   # Add the new domain to the dictionary, and set its value to be a new hyperloglog object
        domains_dict[c_domain].update(c_ip)
        
    return domains_dict[Y].query(X)                    # Returns an approximate count of IP X in domain Y
```


```python
ip_X = '54.27.201.128'       # Here we can change the IP value to be searched for
domain_Y = 'python.org'      # Here we can change the domain to be searched into

approx_count_X_in_Y = Q3_count_IP_X_in_domain_Y(get_stream(), ip_X, domain_Y)

print('Estimated total count of IP "{}" in domain "{}" is: {}'.format(ip_X, domain_Y, approx_count_X_in_Y))
```

    Estimated total count of IP "54.27.201.128" in domain "python.org" is: 8
    

### _Accuracy test: Q3_


```python
true_count_X_in_Y = len(traffic_df[(traffic_df.ip == ip_X) & (traffic_df.domain == domain_Y)])

print('Exact total count of IP "{}" in domain "{}" is: {}'.format(ip_X, domain_Y, true_count_X_in_Y))
```

    Exact total count of IP "54.27.201.128" in domain "python.org" is: 6
    


```python
print('Error difference between CountMinSketch and real-world answer: {}%'
      .format(abs(round(((1 - (approx_count_X_in_Y / true_count_X_in_Y)) * 100),3)))) 
```

    Error difference between CountMinSketch and real-world answer: 33.333%
    

### _Q4: How many unique IPs are there for the domains $d_1$, $d_2$, … ?_


```python
def Q4_unique_ip_joint_count_in_domains(web_traffic_stream, list_of_domains):
    domains_dict = {}                            # Same reasoning for a domain dictionary as in Q2

    for entry in web_traffic_stream:
        c_ip = entry.split('\t')[0]
        c_domain = entry.split('\t')[1]

        if (c_domain in list_of_domains):        # If domain of the entry is within the ones that we are interested in,
            if (not domains_dict.get(c_domain)): # If the key does not yet exist in the domain dictionary, 
                domains_dict[c_domain] = HLL()   # Add the new domain to the dictionary, and set its value to be a new hyperloglog object
                
            domains_dict[c_domain].add(c_ip)     # We update the HLL object of a corresponding domain, by calling its 'add' function upon each IP that visited the domain
            
    joint_hll_obj = None
    
    for domain, hll in domains_dict.items():     # For each individual domain, we will merge its HLL object with the previous ones found
        if (joint_hll_obj == None):
            joint_hll_obj = hll
        joint_hll_obj = joint_hll_obj + hll      # We will merge together the hyperloglog buckets, as implemented in the __add__ function (which also allows us to use the "+" sign for merging)
        
    approx_joint_unique_ip_count = joint_hll_obj.count()
    return approx_joint_unique_ip_count
```


```python
list_of_domains = ['python.org', 'wikipedia.org']      # Here we can change the list of domains to be searched into

approx_joint_unique_ip_count = Q4_unique_ip_joint_count_in_domains(get_stream(), list_of_domains)

print('The approximate number of total unique IPs within the domain list {} is: {}'
.format(list_of_domains, approx_joint_unique_ip_count))
```

    The approximate number of total unique IPs within the domain list ['python.org', 'wikipedia.org'] is: 740003
    

### _Accuracy test: Q4_


```python
true_joint_unique_ip_count = len(traffic_df[traffic_df.domain.isin(list_of_domains) == True].ip.unique())

print('Exact count of the total unique IPs within the domain list {} is: {}'.format(list_of_domains, true_joint_unique_ip_count))
```

    Exact count of the total unique IPs within the domain list ['python.org', 'wikipedia.org'] is: 762628
    


```python
print('Error difference between HLL algorithm and real-world answer: {}%'
      .format(abs(round(((1 - (approx_joint_unique_ip_count / true_joint_unique_ip_count)) * 100),3))))
```

    Error difference between HLL algorithm and real-world answer: 2.967%
    

### _Q5: How many times was IP X seen on domains $d_1$, $d_2$, … ?_


```python
def Q5_joint_count_IP_X_in_domains(web_traffic_stream, X, list_of_domains):
    domains_dict = {}                                      
    
    for entry in web_traffic_stream:
        c_ip = entry.split('\t')[0]
        c_domain = entry.split('\t')[1]

        if (not domains_dict.get(c_domain)): 
            domains_dict[c_domain] = CountMeanMinSketch(w = 49999, d = 10)   
        domains_dict[c_domain].update(c_ip)  
        
    approx_joint_count_of_X = 0
    for domain in list_of_domains:
        approx_joint_count_of_X += domains_dict[domain].query(X)            # sum up the minimum number of counts that IP X was seen in each of the separate domains
    
    return approx_joint_count_of_X
    return approx_joint_count_of_X
```


```python
ip_X = '54.27.201.128'                                           # Here we can change the IP value to be searched for 
list_of_domains = ['python.org', 'wikipedia.org', 'dtu.dk']      # Here we can change the list of domains to be searched into

approx_joint_count_of_X = Q5_joint_count_IP_X_in_domains(get_stream(), ip_X, list_of_domains)

print('Estimated total count of IP "{}" in list of domains {} is: {}'.format(ip_X, list_of_domains, approx_joint_count_of_X))
```

    Estimated total count of IP "54.27.201.128" in list of domains ['python.org', 'wikipedia.org', 'dtu.dk'] is: 10
    

### _Accuracy test: Q5_


```python
true_joint_count_of_X = len(traffic_df[(traffic_df.domain.isin(list_of_domains) == True) & (traffic_df.ip == ip_X)])

print('Exact total count of IP "{}" in list of domains {} is: {}'.format(ip_X, list_of_domains, true_joint_count_of_X))
```

    Exact total count of IP "54.27.201.128" in list of domains ['python.org', 'wikipedia.org', 'dtu.dk'] is: 10
    


```python
print('Error difference between CountMin algorithm and real-world answer: {}%'
      .format(abs(round(((1 - (approx_joint_count_of_X / true_joint_count_of_X)) * 100),3))))
```

    Error difference between CountMin algorithm and real-world answer: 0.0%
    

### _Q6: What are the X most frequent IPs in the stream?_


```python
import heapq

def Q6_X_most_frequent_IPs(web_traffic_stream, X):
    heap = []          # [freq, ip]
    top_k_dict = {}   # ip:[freq, ip]
    
    countmin = CountMinSketch(w = 49999, d = 10)
    
    for entry in web_traffic_stream:
        ip = entry.split('\t')[0]
        countmin.update(ip)
        
        new_freq = countmin.query(ip) 
        
        if ip in top_k_dict:                              # if ip has been recorded in top k dict
            top_k_dict[ip] = new_freq                     # update the dict
            for i in heap:                                # update the heap
                if i[1] == ip:
                    i[0] = new_freq
            heapq.heapify(heap)                           # rebalance the heap
        elif len(top_k_dict) < X:                         # if the heap is not full
            heapq.heappush(heap, [new_freq, ip])
            top_k_dict[ip] = [new_freq, ip]
        elif new_freq > heap[0][0]:                       # if the freq of new ip larger than the root node of the heap
            old_freq = heapq.heappushpop(heap, [new_freq, ip]) # replace
            del top_k_dict[old_freq[1]]
            top_k_dict[ip] = [new_freq, ip]
    X_most = heapq.nlargest(X, heap)
    
    return [x[1] for x in X_most], [x[0] for x in X_most]
```


```python
frequency_X = 10                     # Here we can change the number of X most frequent IPs to search for
approx_X_most_frequent_IPs = Q6_X_most_frequent_IPs(get_stream(), frequency_X)

print('Approximately, the top {} most frequent IPs in the stream are:\n{}'.format(frequency_X, approx_X_most_frequent_IPs[0]))
print('Relevant frequency is: \n{}'.format(approx_X_most_frequent_IPs[1]))
```

    Approximately, the top 10 most frequent IPs in the stream are:
    ['72.187.84.158', '204.42.88.167', '199.104.181.134', '122.109.77.148', '156.145.82.189', '108.41.112.108', '253.174.113.76', '207.104.210.170', '125.88.139.101', '188.221.128.150']
    Relevant frequency is: 
    [248, 236, 234, 232, 225, 220, 217, 213, 213, 211]
    

### _Accuracy test: Q6_


```python
# We will again use our (inefficient) pandas DataFrame object to perform exact calculations on how many unique IPs are there for the above-specified domain list

true_X_most_frequent_IPs = (traffic_df.ip.value_counts().iloc[0:frequency_X].index.tolist())
true_X_most_frequent_freqs = (traffic_df.ip.value_counts().iloc[0:frequency_X].tolist())

print('Truly, the top {} most frequent IPs in the stream are: \n{}'.format(frequency_X, true_X_most_frequent_IPs))
print('Relevant frequency is:\n{}'.format(true_X_most_frequent_freqs))
```

    Truly, the top 10 most frequent IPs in the stream are: 
    ['72.187.84.158', '108.41.112.108', '204.141.72.187', '55.29.199.128', '53.30.199.128', '56.29.200.127', '55.31.199.127', '54.30.199.128', '53.29.198.127', '56.30.200.127']
    Relevant frequency is:
    [222, 199, 139, 34, 34, 33, 32, 31, 30, 30]
    


```python
nr_of_common_elements_between_frequency_lists = len(set(true_X_most_frequent_IPs).intersection(approx_X_most_frequent_IPs[0]))

print('Error difference between countmin sketch using heap algorithm and real-world answer: {}%'
      .format(abs(round(((1 - (nr_of_common_elements_between_frequency_lists / frequency_X)) * 100),3))))
```

    Error difference between countmin sketch using heap algorithm and real-world answer: 80.0%
    
