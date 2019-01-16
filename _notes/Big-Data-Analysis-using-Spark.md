---
layout: note
name: Big Data Analysis using Spark
type: mooc
---

**Aims** - 
1. Introduction to spark/XGBoost/tensorflow - large scale data analysis frameworks
2. Use stats and ML to analyze and identify statistically significant patterns and visually summarize them

**Topics** -
- Memory Hierarchy
- Spark Basics
- Dataframes & SQL
- PCA and weather analysis
- K-means and intrinsic dimensions
- Decision trees, boosting, random forests
- Neural networks & tensorflow

**Memory Hierarchy**
Latency of read-write > latency of computation
Main memory faster than disk

Temporal locality and spatial locality e.g. count words by sorting
Spatial -> memory is stored in blocks not single bytes. Blocks are moved from disk to memory thus the surrounding nearby data is already in memory. Spatial locality makes code faster (x300)

Access locality - cache efficiency. (Try to reuse the same memory elements/blocks repeatedly)
1. Temporal locality - accessing the same elements again and again in quick succession, e.g. theta vector of a neural network
2. Spatial locality - access elements close to each other e.g. indexed array will store elements in same pages/blocks and as soon as the page is copied into the cache, consecutive pairs of array elements will already be cached

Row major vs column major layout - if row major then row-wise traversal is faster than column by column traversal. (e.g. in numpy)

Measuring latency, studying long-tail distributions
Memory latency: time between CPU issuing a command and when the command completes. Time is short if element is already in L1 cache, long if element is in disk/SSD.

Using CDF and loglog plotting, we can see that probability of extreme values (long-tail) is higher than what mean/normal distribution etc would indicate. (Long tail distribution - a distribution far from normal distribution graph line)

Sequential access is much much faster than random access. Writing memory sequentially makes it faster than simple unitary method multiplication logic. Bandwidth vs latency e.g. it takes 8.9s to write 10GB to SSD but that doesn’t imply it takes 8.9 * 10^-10s to write a single byte. Many write ops occur in parallel.
Byte rate for writing memory - 100MB/sec, SSD - 1GB/sec

When reading/writing large blocks, throughput is important, not latency (economies of scale)

Data analytics: moving data around/data access is the bottleneck, NOT computation clock speed

Caching is used to transfer data b/w different levels of memory hierarchy. (Registers/L1/L2/main memory/disk). 
To process big data, we need multiple computers. 

Cluster - collection of many computers connected by ethernet.

![image tooltip here](/images/notes/bigdata/clusters.png)

Abstraction needed to transfer data between computers and hidden from programmers. In Spark, this is the Resilient Distribution Dataset. 

How is data distributed across a cluster? Each file is broken into chunks, (kept track of by chunk master) and each chunk is copied and distributed randomly across machines. 
GFS/HDFS: Break files into chunks, make copies, and distribute randomly among commodity computers in a cluster

-Commodity hardware is used. 

-locality: each data is stored close to a CPU so can do some computation immediately without needing to move data

-redundancy: can recover from hardware failures    

-abstraction: designed to look like a regular filesystem, chunk mechanism is hidden away

MapReduce: Computation abstraction that works well with HDFS. No need to take care of chunks/copies etc

**Spark Basics**

Square each number in a list - 
`[x*x for x in L]`
OR 
`map(lambda x:x*x, L)` [order of elements is not specified]

Sum each number in a list - 
`s = 0; for x in L: s += x`
OR  
`reduce(lambda (x,y): x+y, L)` [order of computation is not specified]

Compute sum of squares of each number in a list - 
S = 0; for x in L: s += x * x
OR
`reduce(lambda (x,y): x+y, map(lambda i:i*i, L))`

Computation order specified vs not specified
Immediate execution vs execution plan
Map/reduce ops should not depend on : order of items in list (commutativity) / order of operations (associativity)
Order independence - result of map/reduce does not depend on order of input. Allows parallel computation, allows compiler to optimize execution plan

Spark is different from Hadoop: stores distributed memory instead of distributed files
Core programming language for Spark: Scala

**Spark Architecture**:

*Spark Context* - connects our software to controller node software that controls the whole cluster. We can control other nodes using the sc object. sc=SparkContext()

*Resilient Distributed Dataset (RDD)*: a list whose elements are distributed over multiple nodes. We may perform parallel operations on this list using special commands, which are based on MapReduce
RDDs are created from a list on the master node or a distributed HDFS file. ‘Collect’ command is used to translate into local list.

```
RDD = sc.parallelize([0,1,2])
RDD.map(lambda x:x*x).reduce(lambda x,y:x+y)
```
^we don’t have direct access to elements of the RDD

```
A = RDD.map(lambda x:x*x) //0,1,2
A.collect() //0,1,4
```

We may take first few elements of a RDD

```
B = sc.parallelize(range(10000))
print B.first
print B.take(5)
print B.sample(False,5/10000).collect() // samples 5 random elements
```

**Master->Slave architecture**

Cluster Master manages the computation resources. Each worker manages a single core. Driver(main() code) runs on the master. (Spatial organization)

![image tooltip here](/images/notes/bigdata/sparkc.png)
 
RDDs are partitioned across the workers. Workers manage these partitions and the executors.
SparkContext is an encapsulation of the cluster for the driver node. Worker nodes communicate with Cluster Manager. 

*Materialization* - intermediate RDDs don’t need to be stored in memory (materialized). 
RDDs by default are not materialized. They are stored if cached - forced calculation of intermediate results.

*Stage* - set of operations that can be done before materialization is necessary i.e. compute one stage, store the result, compute next stage etc (temporal organization)

**Execution concepts** - 

    * RDDs are partitioned across workers
    * RDD graph defines the lineage of RDDs
    * SparkContext divides RDD graph into Stages which define the execution plan
    * a Task is for one stage, for one partition. Executor is the process that performs tasks. Thus tasks are divided by temporal (stage) and spatial (partition) concepts 

Spatial vs Temporal organization : data partitions are stored in different machines vs computation performed in sequence of stages

Plain RDDs: parallelized lists of elements
Types of commands:
1. Creation : RDD from files/databases
2. Transformations: RDD to RDD
3. Actions: RDD to data on driver node/files e.g. .collect(), .count(), .reduce(lambda x:x * x) [sum the elements]

Key-value RDDs: RDDs where elements are a list of (key, value) pairs

```
car_count = sc.parallelize((‘Honda’, 2), (’subaru’, 3), (‘Honda’, 2))
// create RDD
A = sc.parallelize(range(4)).map(lambda x : x*x)
A.collect() // action
B = sc.parallelize([(1,3), (4,5), (-1,3), (4,2)])
B.reduceByKey(lambda x,y: x*y) // each key will now appear only once
 .collect() // [(1,3),(-1,3),(4,10)]
 ```

`groupByKey`: returns a (key, <iterator>) pair for each key. The iterator lists all the values corresponding to the key
`B.groupByKey().map(lambda k,iter: (k, [x for x in iter]))`

`.countByKey()` - returns dictionary with (key, count) pairs for each key
`.lookup(3)` - returns all values for a key
`.collectAsMap()` - like collect() but returns a map instead of a list of tuples. Each key is mapped to the list of values associated with it

Example - find longest and alphabetically last word in a list using Reduce
```
def last_largest(x, y):
    if len(x) > (y): return x
    elif len(y) > (x): return y
    else:
        if x > y: return x
        else: return y

wordRDD.reduce(last_largest)
```

**Programming Assignment 1** -
Q. Given an input file with an arbitrary set of co-ordinates, your task is to use pyspark library functions and write a program in python3 to find if three or more points are collinear.
For instance, if given these points: {(1,1), (0,1), (2,2), (3,3), (0,5), (3,4), (5,6), (0,-3), (-2,-2)}
Result set of collinear points are: {((-2,-2), (1,1), (2,2), (3,3)), ((0,1), (3,4), (5,6)), ((0,-3), (0,1), (0,5))}. Note that the ordering of the points in a set or the order of the sets does not matter.

Solution - 
```
def format_result(x):
    x[1].append(x[0][0])
    return tuple(x[1])

def to_sorted_points(x):
    return tuple(sorted(x))

def to_tuple(x):
    x1 = x.split()
    return (int(x1[0]), int(x1[1]))

def find_slope(x):
    if x[1][0] - x[0][0] == 0:
        return ((x[0], 'inf'), x[1])

    m = (x[1][1] - x[0][1])/(x[1][0] - x[0][0]) # slope = y2 - y1 / x2 - x1
    return ((x[0], m), x[1])

def non_duplicates(x):
    mapping = {}
    for i in range(len(x)):
        if x[i] in mapping:
            return False
        mapping[x[i]] = True
    return True

def get_cartesian(rdd):
    rdd1 = rdd.cartesian(rdd)
    rdd2 = rdd1.filter(non_duplicates)
    return rdd2

def find_collinear(rdd):
    rdd1 = rdd.map(find_slope) # get slope for each pair of points
    rdd2 = rdd1.groupByKey().mapValues(list) # group the values for each key - hash-partition
    rdd3 = rdd2.map(lambda x: format_result(x)).filter(lambda x: len(x) > 2) # format and keep only >2 point lists

    return rdd3

def build_collinear_set(rdd):
    rdd = rdd.map(lambda x: to_tuple(x))
    rdd = get_cartesian(rdd)
    rdd = find_collinear(rdd)
    rdd = rdd.map(to_sorted_points)
    
    return rdd

# test the above methods
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("Collinear Points").setMaster("local[4]") #Initialize spark context using 4 local cores as workers
sc = SparkContext(conf=conf)

from pyspark.rdd import RDD
test_rdd = sc.parallelize(['4 -2', '2 -1', '-3 4', '6 3', '-9 4', '6 -3', '8 -4', '6 9'])
assert isinstance(build_collinear_set(test_rdd), RDD) == True, "build_collinear_set should return an RDD."
```

When we start SparkContext, we can specify the number of workers - usually 1 worker per core. More than 1 worker/core doesn’t increase speed much.

**Spark Internals**

*Lazy Evaluation*: no need to store intermediate results, scan through data only once. e.g. processing 1 million rows with an op that takes 25us doesn’t take 25s. No computation is done immediately, only an execution plan is defined using a dependence graph for how RDDs are computed from each other. We can print the dependence graph using `rdd.toDebugString()`. We may cache (materialize) the results. (`.cache()`). Thus when RDDs are reused, it’s better to materialize them.

![image tooltip here](/images/notes/bigdata/execplan.png)

*Partitions & Gloming*
We can specify no of partitions for a RDD to be distributed into (default - no of workers/cores). Sometimes if the data in diff partitions become unbalanced (e.g. by filtering, some partitions have no data), we have to repartition the RDD.

Glom is used to refer to individual elements. Glom() transforms each partition into a tuple of elements - creates a RDD of tuples, thus can access the elements as lists.
Eg used to count no of elements in each partition - `rdd.glom().map(len).collect()` prints a list like `[20, 43]` etc.