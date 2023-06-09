/**************************************************************************
 * Copyright 2017 Tilo Wiklund, 2022 Johannes Graner
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/

package co.wiklund.disthist

import scala.util.Random
import scala.util.Random._
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashMap._
import scala.collection.mutable.PriorityQueue
import scala.math.{max,floor,round,exp,log,log10,pow}

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.hashing.byteswap32

import org.apache.spark.rdd.{PartitionPruningRDD, RDD}
import org.apache.spark.Partitioner
import org.apache.spark.serializer.JavaSerializer
import org.apache.spark.util.{CollectionsUtils, Utils}

import SubtreePartitionerFunctions._
import BinarySearchFunctions._
import NodeLabelFunctions._
import Types._

import org.apache.spark.RangePartitioner
import org.apache.spark.rdd.RDD

  /**
   * NOTE: Slightly adjusted RangePartitioner Source Code from spark to suite our needs. The idea of estimating subtrees is almost equivalent to distributed sorting;
   *  the only difference being in the last stage when the estimated range bounds are being calculated. We create weighted subtrees and send them to partitions instead of slicing
   *  the data in ranges.
   *
   * SubtreePartitioner - TODO 
   * @param 
   * @param 
   */
class SubtreePartitioner(partitions : Int, rdd : RDD[(NodeLabel, Count)], val samplePointsPerPartitionHint : Int = 20) extends Partitioner {
  
   // We allow partitions = 0, which happens when sorting an empty RDD under the default settings.
  require(partitions >= 0, s"Number of partitions cannot be negative but found $partitions.")
  require(samplePointsPerPartitionHint > 0,
    s"Sample points per partition must be greater than 0 but found $samplePointsPerPartitionHint")

  implicit val ordering: Ordering[NodeLabel] = leftRightOrd

  val subtrees : Array[(NodeLabel, Float)] = {
    if (partitions <= 1) {
      /*TODO FIX so that it works for 1 partition */
      Array((rootLabel, 1))
    } else {
      // This is the sample size we need to have roughly balanced output partitions, capped at 1M.
      // Cast to double to avoid overflowing ints or longs
      val sampleSize = math.min(samplePointsPerPartitionHint.toFloat * partitions, 1e6)
      // Assume the input partitions are roughly balanced and over-sample a little bit.
      val sampleSizePerPartition = math.ceil(3.0 * sampleSize / rdd.partitions.length).toInt
      //TODO
      val (numItems, sketched) = sketch(rdd.map(_._1), sampleSizePerPartition)
      if (numItems == 0L) {
        throw new IllegalArgumentException("Could not sample any points from partition, is it empty?")
      } else {
        // If a partition contains much more than the average number of items, we re-sample from it
        // to ensure that enough items are collected from that partition.
        val fraction = math.min(sampleSize / math.max(numItems, 1L), 1.0)
        val candidates = ArrayBuffer.empty[(NodeLabel, Float)]
        val imbalancedPartitions = mutable.Set.empty[Int]
        sketched.foreach { case (idx, n, sample) =>
          if (fraction * n > sampleSizePerPartition) {
            imbalancedPartitions += idx
          } else {
            // The weight is 1 over the sampling probability.
            val weight = (n.toFloat / sample.length).toFloat
            for (key <- sample) {
              candidates += ((key, weight))
            }
          }
        }
        if (imbalancedPartitions.nonEmpty) {
          // Re-sample imbalanced partitions with the desired sampling probability.
          val imbalanced = new PartitionPruningRDD(rdd.map(_._1), imbalancedPartitions.contains)
          val seed = byteswap32(-rdd.id - 1)
          //TODO
          val reSampled = imbalanced.sample(withReplacement = false, fraction, seed).collect()
          val weight = (1.0 / fraction).toFloat
          candidates ++= reSampled.map(x => (x, weight))
        }
        val weightLimit : Float = candidates.length.toFloat / partitions
        val maxSubtrees = maximalWeightSubtreeGeneration(candidates.sorted.toVector, weightLimit)
        maxSubtrees
      }
    }
  }

  /* maximal depth for all subtrees */
  val maxSubtreeDepth : Depth = {
    var maxDepth : Depth = 0 
    subtrees.foreach(subtree => {
      (maxDepth < subtree._1.depth) match {
        case true => { maxDepth = subtree._1.depth }
        case false  => { maxDepth = maxDepth }
      }
    })
    maxDepth
  }

  val subtreePartitionMap = 
    distributeSubtreesToPartitions(partitions, subtrees)

  override def equals(other: Any): Boolean = other match {
    case r : SubtreePartitioner =>
      r.subtrees.sameElements(subtrees)
    case _ =>
      false
  }

  def numPartitions: Int = math.min(partitions, subtrees.length) 

  override def getPartition(key : Any) : Int = {
    val leaf = key.asInstanceOf[NodeLabel]
    subtreePartitionMap(findSubtree(leaf, subtrees))
  }
}

object SubtreePartitionerFunctions {

  /**
   * Sketches the input RDD via reservoir sampling on each partition.
   *
   * @param rdd the input RDD to sketch
   * @param sampleSizePerPartition max sample size per partition
   * @return (total number of items, an array of (partitionId, number of items, sample))
   */
  def sketch(
      rdd: RDD[NodeLabel],
      sampleSizePerPartition: Int): (Long, Array[(Int, Long, Array[NodeLabel])]) = {
    val shift = rdd.id
    // val classTagK = classTag[K] // to avoid serializing the entire partitioner object
    val sketched = rdd.mapPartitionsWithIndex { (idx, iter) =>
      val seed = byteswap32(idx ^ (shift << 16))
      val (sample, n) = reservoirSamplingLAndCount(iter, sampleSizePerPartition, seed)
      Iterator((idx, n, sample))
    }.collect()
    val numItems = sketched.map(_._2).sum
    (numItems, sketched)
  }

 /**
  * reservoirSamplingLAndCount - Implementation of Reservoir Sampling algorithm L, [Reservoir-sampling algorithms of time complexity O(n(1+log(N/n))), Kim-Hung Li].
  * @param iter - The iterator over the leaves from which we wish to sample
  * @param sampleSize - The sample size
  * @param seed - The seed 
  * @return an iterator containing the generated sample and the length of the input
  * */
  def reservoirSamplingLAndCount(iter : Iterator[NodeLabel], sampleSize : Int, seed : Long = Random.nextLong()) : (Array[NodeLabel], Long) = {
    val random = new Random(seed)
    var sample : Array[NodeLabel] = new Array(sampleSize)
    var len : Long = 0
    var i = 0
    while (i < sampleSize) {
      if (!iter.hasNext) {
        sample = sample.take(i)
        i = sampleSize
      } else {
        sample(i) = iter.next
        i += 1
        len += 1
      }
    }

    var W : Double = 1.0
    var continue = true
    var rem = iter
    while (continue) {
      W = W * exp(log(random.nextDouble)/sampleSize)
      val S = floor(log(random.nextDouble)/log(1-W)).toInt
      rem = rem.drop(S)
      len += S
      if (rem.hasNext) {
        sample(random.nextInt(sampleSize)) = rem.next 
        len += 1
      } else {
        continue = false 
      }
    }
    
    (sample, len)
  }

  /**
   * maximalSubtreesDecreaseDepth - Decreases the subtrees' depths' as much as possible without having them
   *                                intersect each other.
   * @param subtrees - The vector of maximal non-intersecting subtrees
   *
   * @assumptions: Assumes the maximal subtrees to be ordered from left to right.
   */
  def maximalSubtreesDecreaseDepth[A : ClassTag](maxSubtrees : Array[(NodeLabel, A)], depthLimit : Depth = 0) : Array[(NodeLabel,A)] = {
    val maxLen = maxSubtrees.length
    require(maxLen > 0)

    if (maxLen == 1) {
      Array((rootLabel, maxSubtrees(0)._2))
    } else {

      val subtreeFirst = {
        val maxAnc = join(maxSubtrees(0)._1, maxSubtrees(1)._1).left
        (depthLimit <= maxAnc.depth) match {
          case true =>  { (maxAnc, maxSubtrees(0)._2) } 
          case false => { (maxSubtrees(0)._1.truncate(depthLimit), maxSubtrees(0)._2) } 
        }
      }
      var finalSubtrees = List(subtreeFirst)

      for (i <- 1 to (maxLen-2)) {
        val prevTree = maxSubtrees(i-1)
        val curTree = maxSubtrees(i)
        val nextTree = maxSubtrees(i+1)
        val subtreePossibility1 = join(prevTree._1, curTree._1).right
        val subtreePossibility2 = join(curTree._1, nextTree._1).left

        /**
         * The maximum subtree of curTree is the largest subtree containing curTree while 
         * not crossing the previous and the next trees' paths. If the subtree is over the depth limit
         * (by default 0), increase the depth to the depth limit
         */
        val maxAnc = (subtreePossibility1.depth < subtreePossibility2.depth) match {
          case true  => { subtreePossibility2 }
          case false => { subtreePossibility1 }
        }

        val maxTree = (depthLimit <= maxAnc.depth) match {
          case true =>  { (maxAnc, curTree._2) } 
          case false => { (maxSubtrees(i)._1.truncate(depthLimit), curTree._2) } 
        }

        finalSubtrees = List(maxTree) ::: finalSubtrees
      }

      val maxAnc = join(maxSubtrees(maxLen-2)._1, maxSubtrees(maxLen-1)._1).right
      val subtreeLast = (depthLimit <= maxAnc.depth) match {
        case true =>  { (maxAnc, maxSubtrees(maxLen-1)._2) } 
        case false => { (maxSubtrees(maxLen-1)._1.truncate(depthLimit), maxSubtrees(maxLen-1)._2) } 
      }

      finalSubtrees = List(subtreeLast) ::: finalSubtrees
      finalSubtrees.toArray.reverse
    }
  }

  def subtreeAt(at : NodeLabel, leaves : Vector[(NodeLabel,Float)]) : Subset = {
    val within = Subset(0, leaves.length)
    val low =  binarySearchWithin((x : (NodeLabel, Float)) => !isStrictLeftOf(x._1, at))(leaves, within)
    val high = binarySearchWithin((x : (NodeLabel, Float)) => x._1 != at && !isDescendantOf(x._1, at))(leaves, within.sliceLow(low))
    Subset(low, high).normalise
  }


  /**
   * TODO: REWRTIE (LIKE COUNT IN MERGE)
   * maximalWeightSubtreeGeneration - Find the set of largest non-intersecting subtrees of the tree
   * @param leaves The tree of sorted leaves (leftRightOrd) determining the tree, each leaf containing a weight which corresponds to how many elements in the data it represents
   * @param weightLimit maximal weight of subtrees 
   */
  def maximalWeightSubtreeGeneration(leaves : Vector[(NodeLabel, Float)], weightLimit: Float) : Array[(NodeLabel, Float)] = {
    var maxSubtrees : List[(NodeLabel, Float)] = List.empty
    val len = leaves.length
    require(len > 0) 

    var oldMaxIndex : Int = -1 /* last generated subtree's max leaf index */
    var subtree : (NodeLabel, Float) = leaves(0)
    var i : Int = 1

    while (i < len) {
      val commonAncestor = join(subtree._1, leaves(i)._1)
      val ancestorInterval : Subset = subtreeAt(commonAncestor, leaves)

      var ancestorWeight = subtree._2
      for (j <- i until ancestorInterval.upper) {
        ancestorWeight += leaves(j)._2
      }

      if (ancestorWeight <= weightLimit && oldMaxIndex < ancestorInterval.lower) {
        subtree = (commonAncestor, ancestorWeight)
        i = ancestorInterval.upper
        /**
         * Special case: There are some many sampled duplicates than they cannot fit in a whole subtree, so
         *  we must force them to fit 
         **/
      } else if (leaves(i)._1 == commonAncestor) {
        subtree = (commonAncestor, ancestorWeight)
        if (ancestorInterval.upper < len) {
          maxSubtrees = List(subtree) ::: maxSubtrees
          subtree = leaves(ancestorInterval.upper) 
          oldMaxIndex = ancestorInterval.upper - 1 
          i = ancestorInterval.upper + 1
        } else {
          i = len;
        }
      } else {
        maxSubtrees = List(subtree) ::: maxSubtrees
        subtree = leaves(i) 
        oldMaxIndex = i-1
        i += 1
      }
    }
    maxSubtrees = List(subtree) ::: maxSubtrees

    /**
     * maxSubtrees now contain the largest subtrees with regards to their weights,
     * exluding 0 weight subtrees, i.e. we have split the leaves into maximal subtrees
     * under the count limit. We now have to decrease the depth of all generated subtrees
     * up until they start to cross eachother's root path. 
     */
    maximalSubtreesDecreaseDepth[Float](maxSubtrees.toArray.reverse)
  }

  /**
   * subtreeDistributeToPartittions - Distribute subtrees over the partitions according to a count priority.
   *                                 The returned map is a mapping of subtrees to partitions. The returned
   *                                 integer is the partition with the smallest total count.
   * @param numPartitions - The number of partitions to distribute subtrees over
   * @param subtrees - The maximal leftRightOrdered subtrees to distribute
   */
  def distributeSubtreesToPartitions(numPartitions : Int, subtrees: Array[(NodeLabel, Float)]) : HashMap[NodeLabel, Int] = {

    object subtreeWeightOrdering extends Ordering[(NodeLabel, Float)] {
      def compare(a : (NodeLabel, Float), b : (NodeLabel, Float)) = {
        if (a._2 == b._2) 0 else if (a._2 < b._2) -1 else 1
      }
    }
    val subtreeWeightSorted = subtrees.sorted(subtreeWeightOrdering.reverse)

    object minPriorityOrdering extends Ordering[(Float, Int)] {
      def compare(a : (Float, Int), b : (Float, Int)) = {
        if (a._1 == b._1) 0 else if (a._1 > b._1) -1 else 1
      }
    }
    var partitionQueue : PriorityQueue[(Float, Int)] = new PriorityQueue()(minPriorityOrdering) 
    var subtreePartitionMap : HashMap[NodeLabel, Int] = new HashMap()

    (0 until numPartitions).map(_ => 0.0f).zipWithIndex.foreach(partitionQueue.enqueue(_))
    subtreeWeightSorted.foreach(subtree => {
        val lowestWeightPartition = partitionQueue.dequeue
        partitionQueue.enqueue((lowestWeightPartition._1 + subtree._2, lowestWeightPartition._2))
        subtreePartitionMap += subtree._1 -> lowestWeightPartition._2
    })

    println("---- PARTITION SAMPLE DISTRIBUTION ----")
    partitionQueue.foreach(t => println("(partition,weight): (" + t._2 + "," + t._1 + ")"))

    subtreePartitionMap
  }

  /**
   * findSubtree - Find the leaf's subtree within the maximalSubtree Array, if one exists, otherwise, return
   *               the closest subtree found.
   * @param leaf - The leaf node 
   * @param subtree - The subtree vector to search within
   */
  def findSubtree(leaf : NodeLabel, subtrees : Array[(NodeLabel, Float)]) : NodeLabel = {
    var low = 0
    var high = subtrees.length-1
    var midIndex : Int = -1
    while (low <= high) {
      midIndex = low + ((high-low) - ((high-low) % 2)) / 2
      val mid = subtrees(midIndex)

      /* Valid, one will never be the ancestor of the other, assuming splitting to some finest resolution... */
      leftRightOrd.compare(leaf.truncate(mid._1.depth), mid._1) match { 
        case  0 => { return mid._1       }
        case -1 => { high = midIndex - 1 } /* leaf's subtree must be left of mid  */
        case  1 => { low  = midIndex + 1 } /* leaf's subtree must be right of mid */
      }
    }
  
    /**
     * leaf has no ancestor and is left of mid, so we check the subtrees 
     * (high, midIndex), or just midIndex if high does not exist. Otherwise
     * do a similar analysis for (midIndex, low)
     */
    var closestSubtree : NodeLabel = subtrees(midIndex)._1
    if (high < midIndex && high >= 0) {
      /* Check ancestor depth between high, leaf,  and mid, leaf */
      val highTree : NodeLabel = subtrees(high)._1
      join(leaf, closestSubtree).depth < join(leaf, highTree).depth match {
        case true => { closestSubtree = highTree }
        case false => { closestSubtree = closestSubtree }
      }
    } else if (low <= subtrees.length-1) {
      /* Check ancestor depth between low, leaf,  and mid, leaf */
      val lowTree : NodeLabel = subtrees(low)._1
      join(leaf, closestSubtree).depth < join(leaf, lowTree).depth match {
        case true =>  { closestSubtree = lowTree }
        case false => { closestSubtree = closestSubtree }
      }
    } 
    closestSubtree
  }

  def findSubtree(leaf : NodeLabel, subtrees : Vector[NodeLabel]) : NodeLabel = {
    var low = 0
    var high = subtrees.length-1
    var midIndex : Int = -1
    while (low <= high) {
      midIndex = low + ((high-low) - ((high-low) % 2)) / 2
      val mid = subtrees(midIndex)

      /* Valid, one will never be the ancestor of the other, assuming splitting to some finest resolution... */
      leftRightOrd.compare(leaf.truncate(mid.depth), mid) match { 
        case  0 => { return mid          }
        case -1 => { high = midIndex - 1 } /* leaf's subtree must be left of mid  */
        case  1 => { low  = midIndex + 1 } /* leaf's subtree must be right of mid */
      }
    }
  
    /**
     * leaf has no ancestor and is left of mid, so we check the subtrees 
     * (high, midIndex), or just midIndex if high does not exist. Otherwise
     * do a similar analysis for (midIndex, low)
     */
    var closestSubtree : NodeLabel = subtrees(midIndex)
    if (high < midIndex && high >= 0) {
      /* Check ancestor depth between high, leaf,  and mid, leaf */
      val highTree : NodeLabel = subtrees(high)
      join(leaf, closestSubtree).depth < join(leaf, highTree).depth match {
        case true => { closestSubtree = highTree }
        case false => { closestSubtree = closestSubtree }
      }
    } else if (low <= subtrees.length-1) {
      /* Check ancestor depth between low, leaf,  and mid, leaf */
      val lowTree : NodeLabel = subtrees(low)
      join(leaf, closestSubtree).depth < join(leaf, lowTree).depth match {
        case true =>  { closestSubtree = lowTree }
        case false => { closestSubtree = closestSubtree }
      }
    } 
    closestSubtree
  }
}
