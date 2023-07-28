/**************************************************************************
 * Copyright 2022 Johannes Graner
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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.util.LongAccumulator

import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.P

import Types._
import LeafMapFunctions._
import HistogramUtilityFunctions._
import BinarySearchFunctions._
import NodeLabelFunctions._
import SubtreePartitionerFunctions._

object MergeEstimatorFunctions {

  private def getSpark: SparkSession = {
    val sparkOpt = org.apache.spark.sql.SparkSession.getActiveSession
    if (sparkOpt.isEmpty) throw new UnsupportedOperationException("Spark is not running.")
    sparkOpt.get
  }

  def labelAtDepth(tree: SpatialTree, depth: Int, points: RDD[MLVector]): RDD[(BigInt, MLVector)] =
    points.map(x => (tree.descendBox(x)(depth).lab, x))

  def labelAtDepth(tree: SpatialTree, depth: Int, points: Dataset[Array[Double]]): Dataset[(NodeLabel, Array[Double])] = {
    val spark = getSpark
    import spark.implicits._
    points.map(x => (tree.descendBox(Vectors.dense(x))(depth), x))
  }

  def labeledToCountedDS(labeledRDD: RDD[(BigInt, MLVector)]): Dataset[(NodeLabel, Count)] = {
    val spark = getSpark
    import spark.implicits._
    labeledRDD
      .map{ case (lab, vec) => NodeLabel(lab) }
      .toDS.groupByKey(node => node).count
  }

  def labeledToCountedDS(labeledDS: Dataset[(NodeLabel, Array[Double])]): Dataset[(NodeLabel, Count)] = {
    val spark = getSpark
    import spark.implicits._
    labeledDS.groupByKey(_._1).count
  }

  /**
   * quickToLabeled - Quick labeling function of datapoints.
   * 
   * @param tree - Root Regular Paving 
   * @param depth - The depth to split down to and determine the label
   * @param points - The points to label and count
   * @return A RDD containing cell NodeLabels for cells with non-zero counts. Every NodeLabel occurs only once
   *         with the count set as the number of points found within the cell.
   */
  def quickToLabeled(tree: WidestSplitTree, depth: Int, points: RDD[MLVector]): RDD[(NodeLabel, Count)] = {
    val spark = getSpark
    import spark.implicits._

    require(depth > 0)
    points.mapPartitions(iter => tree.quickDescend(iter, depth)).map(n => (n,1L)).reduceByKey((v1, v2) => v1 + v2)
  }

  def getTree[A](tree: SpatialTree)(f: SpatialTree => A) = f(tree)

  def mergeLeavesStep(
    tree: SpatialTree, 
    countedDS: Dataset[(NodeLabel, Count)], 
    countLimit: Count, 
    truncDepth: Int
  ): Dataset[(NodeLabel, Count)] = getTree(tree) { tree =>
    val spark = getSpark
    import spark.implicits._

    def mergeHistograms(
      ancestor: NodeLabel, 
      nodesAndCounts: Iterator[(NodeLabel, Count)]
    ): Iterable[(NodeLabel, Count)] = {
      val seq = nodesAndCounts.toSeq
      val count = seq.unzip._2.sum
      val mergeAll = count <= countLimit
      if (mergeAll)
        Seq((ancestor, count))
      else {
        val hist = Histogram(tree, count, fromNodeLabelMap(seq.toMap))
        val rootHist = Histogram(tree, count, LeafMap(Truncation(Vector(ancestor)), Vector(count)))
        val prio: PriorityFunction[Count] = (_, c, _) => c
        val backTrackedHist = hist
          .backtrackTo(prio, rootHist)
          .takeWhile(h => h.counts.vals.forall(_ <= countLimit))
          .lastOption
          .getOrElse(hist)
        backTrackedHist.counts.toIterable
      }
    }

    countedDS.groupByKey{ case (node, _) => 
      node.truncate(truncDepth)
    }.flatMapGroups(mergeHistograms)
  }

  def mergeLeaves(
    tree: SpatialTree,
    countedDS: Dataset[(NodeLabel, Count)],
    countLimit: Count,
    bulkStepSize: Int,
    checkpointPath: String,
    verbose: Boolean = false
  ): Dataset[(NodeLabel, Count)] = {
    val spark = getSpark
    import spark.implicits._

    var stepSize = bulkStepSize
    var tmpDS = countedDS.cache
    var currentDepth = tmpDS.first._1.depth
    val initialCount = tmpDS.count
    var lastCount = initialCount
    while (currentDepth > 1 && (tmpDS.count < lastCount || lastCount == initialCount)) {
      while (currentDepth <= stepSize) stepSize /= 2

      lastCount = tmpDS.count
      currentDepth -= stepSize
      if (verbose) println(s"Merging at depth ${currentDepth}")

      val mergedDS = mergeLeavesStep(tree, tmpDS, countLimit, currentDepth)
      mergedDS.write.mode("overwrite").parquet(checkpointPath)

      tmpDS.unpersist
      tmpDS = spark.read.parquet(checkpointPath).as[(NodeLabel, Count)].cache
      if (verbose) println(s"last count is $lastCount and current count is ${tmpDS.count}")
    }
    tmpDS
  }

  def collectHistogram(tree: SpatialTree, ds: Dataset[(NodeLabel, Count)]): Histogram = {
    val leafMap = fromNodeLabelMap(ds.collect.toMap)
    Histogram(tree, leafMap.vals.sum, leafMap)
  }

  /**
   * subtreeAt - Find the lower (inclusive) and upper (exclusive) indices of the leaves in the subtree of the given node with
   *             label at. 
   *
   * @param at - The Nodelabel of the subtree's root.
   * @param leaves - The vector of sorted leaves
   * @return Subset containing two integer indices lower (inclusive) and upper (exclusive). 
   *
   * @assumption The vector should be left-right ordered
   */
 def subtreeAt(at : NodeLabel, leaves : Vector[(NodeLabel,Count)]) : Subset = {
    val within = Subset(0, leaves.length)
    val low =  binarySearchWithin((x : (NodeLabel, Count)) => !isStrictLeftOf(x._1, at))(leaves, within)
    val high = binarySearchWithin((x : (NodeLabel, Count)) => x._1 != at && !isDescendantOf(x._1, at))(leaves, within.sliceLow(low))
    Subset(low, high).normalise
  }

  /**
   * maximalCountSubtreeGeneration - Generates the set of maximal subtrees, from left to right, with count not exceeding the count
   *                                 limit or the given depth limit. This means that the subtrees cannot be decreased in depth 
   *                                 without intersecting another subtree, or going below the depth limit, or exceeding the count limit. 
   *                                 If the depth limit is reached within the function, the function reports back through an accumulator
   *                                 that the limit was reached, and that the merging will not be able to finish fully within this worker's
   *                                 function call.
   *
   * @param leavesIter - Leaves to generate subtrees from
   * @param mergeShouldContinue - Accumulator to signal if more merging need to take place outside of function
   * @param countLimit - the count limit to merge up to
   * @param depthLimit - the depth limit every worker must adhere to
   * @return Iterator over left-to-right ordered fully or partially merged leaves
   *
   * @assumption The input iterator of leaves is left-right ordered 
   */
  def maximalCountSubtreeGeneration(leavesIter : Iterator[(NodeLabel, Count)], mergeShouldContinue : LongAccumulator, countLimit : Count, depthLimit : Depth) : Iterator[(NodeLabel, Count)] = {

    var maxSubtrees : List[(NodeLabel, Count)] = List.empty
    val leaves = leavesIter.toVector
    val len = leaves.length
    require(len > 0) 

    var oldMaxIndex : Int = -1 /* last generated subtree's max leaf index */
    var subtree : (NodeLabel, Count) = (leaves(0)._1, leaves(0)._2)
    var subtreeCount : Count = leaves(0)._2 
    var beginNewSubtree : Boolean = false 
    var i : Int = 1

    while (i < len) {
      val commonAncestor = join(subtree._1, leaves(i)._1)
      if (depthLimit <= commonAncestor.depth) {

        val ancestorInterval : Subset = subtreeAt(commonAncestor, leaves)
        var ancestorCount = subtree._2
        /* early exist, note that each bin >= 1 count so cannot iterate over the countLimit */
        if (ancestorCount + ancestorInterval.upper - i <= countLimit) {
          for (j <- i until ancestorInterval.upper) {
            ancestorCount += leaves(j)._2
          }
        } else {
          ancestorCount = countLimit + 1
        }

        if (ancestorCount <= countLimit && oldMaxIndex < ancestorInterval.lower) {
          subtree = (commonAncestor, ancestorCount)
          i = ancestorInterval.upper
        } else if (ancestorCount > countLimit && subtree._1 == commonAncestor) {
          throw new IllegalArgumentException("A bin had a larger count than the countLimit at the start of merging") 
        } else {
          beginNewSubtree = true
        }
      } else {
        mergeShouldContinue.add(1)
        beginNewSubtree = true
      }

      if (beginNewSubtree) {
        beginNewSubtree = false 
        maxSubtrees = List(subtree) ::: maxSubtrees
        subtree = (leaves(i)._1, leaves(i)._2) 
        if (subtree._2 > countLimit) {
          throw new IllegalArgumentException("A leaf had exceeded the countLimit from the start, consider splitting more") 
        }
        oldMaxIndex = i-1
        i += 1
      }
    }

    maxSubtrees = List(subtree) ::: maxSubtrees
    maximalSubtreesDecreaseDepth[Count](maxSubtrees.toArray.reverse, depthLimit).toIterator
  }

  /**
   * mergeLeavesRDD - Merge the leaves or cells up to the count limit and return the RDD of the generated leaves.
   *
   * @param tree - The root Regular Paving
   * @param countedRDD - The set of cells with Labels and non-zero counts
   * @param countLimit - The maximum count for any cell to have
   * @param depthLimit - The depth limit is the depth for which it is safe for every worker to locally merge leaves up to.
   *                     If further merging has to be done, the results are sent to the driver and finished locally.
   * @param verbose - Verbose printing of how the operation is going
   *
   * @return The RDD in which every leaf contain a count not exceeding the given count limit but any more merging of leaves
   *         make the new leaf exceed the count limit.
   */
  def mergeLeavesRDD(countedRDD: RDD[(NodeLabel, Count)], countLimit: Count, depthLimit : Depth, verbose: Boolean = false): Array[(NodeLabel, Count)] = {
    val spark = getSpark
    import spark.implicits._

    if (verbose) { println("--- RDD MERGING START ---") }
    
    /* If merging of RDDs reached the depthLimit, continue the last merges on driver */
    val mergeShouldContinue = new LongAccumulator
    val acc = spark.sparkContext.register(mergeShouldContinue)

    /* Merge up to countLimit or minDepth for each partition */
    var subtreesMerged = countedRDD.mapPartitions(iter => maximalCountSubtreeGeneration(iter, mergeShouldContinue, countLimit, depthLimit), preservesPartitioning = true).collect.sortBy(t => t._1)(leftRightOrd)

    if (verbose) { 
      println("--- RDD COLLECTED AFTER MERGE  ---") 
    }

    if (!mergeShouldContinue.isZero) {
      /* Ignore Accumulator this time, depthLimit is 0 */
      subtreesMerged = maximalCountSubtreeGeneration(subtreesMerged.toIterator, mergeShouldContinue, countLimit, 0).toArray
    }

    if (verbose) { 
      println("--- RDD MERGING DONE ---")
    }

    subtreesMerged
  }

  /**
   * mergeLeavesHistogram - Merge the leaves or cells up to the count limit and return the Histogram.
   *
   * @param tree - The root Regular Paving
   * @param countedRDD - The set of cells with Labels and non-zero counts
   * @param countLimit - The maximum count for any cell to have
   * @param depthLimit - The depth limit is the depth for which it is safe for every worker to locally merge leaves up to.
   *                     If further merging has to be done, the results are sent to the driver and finished locally.
   * @param verbose - Verbose printing of how the operation is going
   *
   * @return The coarsest histogram in which every leaf contain a count not exceeding the given count limit.
   */
  def mergeLeavesHistogram(tree : SpatialTree, countedRDD: RDD[(NodeLabel, Count)], countLimit: Count, depthLimit : Depth, verbose: Boolean = false): Histogram = {
    val merged = mergeLeavesRDD(countedRDD, countLimit, depthLimit, verbose)
    Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap))
  }
}
