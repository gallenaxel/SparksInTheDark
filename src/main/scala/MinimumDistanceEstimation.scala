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

/**
 * ---------------------- Distributed Scheffe Set Calculations ---------------------
 *
 * We start by noting that calculating the following part in Delta_theta is not hard!
 * 
 * 	INTEGRAL[A_(phi,eta)] f_theta dx,   A_(phi,eta) := Sheffe set {f_phi > f_eta}.
 * 
 * For any chosen K histograms, this requires O(K^2 * n) computations, n = |CRP.size|
 * since any such A is the union of a set of leaves L_i. Thus, whenever we pass over
 * a leaf L_i in our linear traversal of all leaves in the CRP, we simply note which
 * Scheffe sets the leaf belongs to, and calculate
 * 
 * 	INTEGRAL[L_i] f_theta dx
 * 
 * Since all leaves are disjoint, we have that 
 * 	
 * 	INTEGRAL[A(phi,eta)] f_theta dx 
 * 		= INTEGRAL[Union_i L_i] f_theta dx
 * 		= SUM_i [INTEGRAL[L_i] f_theta dx]
 * 	
 * Therefore, the calculation of the integral can be done fully distributed in a
 * map-reduce operation, where every worker can calculate all the local intergrals
 * of its leaves (of the finest histogram) and the sum of the final sums from each 
 * worker. This will probably only affect the performance of the MDE method if the 
 * histogram is really large; so it might not be to important to make it distributed. 
 * Nonetheless, it can be done fast, and also seperated from the larger performance
 * hog; namely the calculations of the empirical measure over A_(phi,eta).
 * 
 * The biggest win we can achieve here is to remove worker communications + improve
 * the time complexity for creating the first validation (leaves,counts) at the CRP
 * -depth + reusing created validation (leaves, counts):
 * 
 * (1) First iteration pass of computational complexity O(k^2 * v) where v is the length
 *     of the validation (leaves, counts).
 * 
 * (2) In any successive iteration, use the validation (leaves, counts) generated at 
 *     the CRP-depth in the previous iteration as an intermediate validation 
 *     (leaves,counts), instead of starting over from the beginning with the original
 *     validation (leaves,counts).
 * 
 * This will only require us to apply a sort on locally within the validationRDD
 * partitions as a pre-processing step for the algorithm. Then, only the first iteration
 * (1) should be of any larger cost while iterations in (2) should be super quick.
 * By applying these array iteration algorithms, we only have to store the finest 
 * histogram we are considering, and for any RDD data, we can make the partitions 
 * as refined as need be so that the can be stored in main-memory as well. The original
 * partitions should work though, since we could apply the splits and hold the 
 * partitions in memory at the start of the estimation procedure.
 * 
 * In the algorithm, which we describe below, each worker would pass through, from left
 * to right, both the validation leaves (which it owns) and the whole finest histogram.
 * Thus we can in the same algorithm divide up the work of calculating density integrals
 * over Scheffe sets and calculating the empirical measure values! Since the empirical
 * measure's value is simply
 * 
 * 	 #{ A_(phi,eta) count }     SUM_T #{ Worker T's count in A_(phi,eta) }
 * 	------------------------ = --------------------------------------------.
 * 	 #{ Validation count }               #{ Validation count }
 * 
 * Again this is simply a map-reduce operation, similar to the first sum of integrals.
 * If we for the moment skip the distribution of the integral calculation stuff, and
 * instead focus on the empirical measure, the algorithm becomes as follows:
 * 
 * Algorithm(finestHistogram : CRP, iterator : LeavesIterator)
 * {
 * (0)	leaves <- Either prev.  + counts or the partitoned and sorted original RDD.
 * (1)	hi <- Initalize histogramIndex to start of finestHistogram leaves (sorted)
 * (2) 	newLeaves <- (finestHistogram.leaves, 0)
 * (3) 	countMatrix <- MATRIX[K,K] = 0
 * 	FOR (leaf in iterator) DO 
 * 		WHILE (histogram.leaves(hi) != leaf.ancestor) DO 
 * (4)			hi += 1
 * 		END
 * (5)		newLeaves(hi).count += leaf.count	
 * 		FOR (i = 0 UNTIL K)
 * 			FOR (j = i+1 UNTIL K)	
 * (6)				IF (leaf[ f_(ij) ] > leaf[ f_(ji) ]) DO
 * 					countMatrix[i][j] += leaf.count
 * 				ELSE IF (leaf[ f_(ij) ] < leaf[ f_(ji) ])
 * 					countMatrix[j][i] += leaf.count
 * 				END
 * 			END
 * 		END
 * 	END
 * (7)	return (newLeaves, countMatrix) 
 * }
 */

package co.wiklund.disthist

import math.min

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{ SparkSession }
import org.apache.spark.rdd.RDD

import Types._
import HistogramUtilityFunctions._
import HistogramFunctions._
import LeafMapFunctions._
import SubtreePartitionerFunctions._
import NodeLabelFunctions._

object MDEFunctions {

  private def getSpark: SparkSession = {
    val sparkOpt = org.apache.spark.sql.SparkSession.getActiveSession
    if (sparkOpt.isEmpty) throw new IllegalArgumentException("Make sure Spark is running!")
    sparkOpt.get
  }

  /**
   * spacedBacktrack - Backtrack from finest histogram to evenly spaced coarser histograms which are to be considered in the iteration's MDE calculations. 
   *
   * @param hist - The finest histogram under consideration
   * @param startIndex - The new finest histogram's index relative to hist
   * @param stopIndex - The most coarsest histogram's index relative to hist under consideration
   * @param stepSize - The number of splits between the generated histograms
   * @param verbose - Verbose printing of the process
   * @param prio - The priority function used in backtracking the histogram
   *
   * @return A set of evenly spaced histograms along the search path.
   */
  def spacedBacktrack(
    hist: Histogram, 
    startIndex: Int, 
    stopIndex: Int, 
    stepSize: Int, 
    verbose: Boolean = false, 
    prio: PriorityFunction[Count] = {case (_, c, _) => c }
  ): Vector[Histogram] = {
    //val backtrackedSpaced = hist.backtrack(prio).zipWithIndex.drop(startIndex).filter{ case (_, i) => (i - startIndex) % stepSize == 0}.takeWhile(_._2 <= stopIndex).map(_._1)

    var startHist = startIndex match {
      case 0 => hist
      case _ => hist.backtrackNumSteps(prio, startIndex)
    }

    var index = startIndex
    var numBacktracks = 0
    /* May not go furter than stopindex, but must also not merge further than root */
    val lim = min(stopIndex, startHist.counts.vals.length-1)
    while (index + (numBacktracks + 1) * stepSize <= lim) {
      numBacktracks += 1
    }

    var backtrackedSpaced : Array[Histogram] = new Array(numBacktracks + 1)
    backtrackedSpaced(0) = hist
    for (i <- 1 until backtrackedSpaced.length) {
      backtrackedSpaced(i) = backtrackedSpaced(i-1).backtrackNumSteps(prio, stepSize)
    }
 
    if (verbose) {
      val initialSize = hist.counts.leaves.length
      println(s"Backtracking from histogram of size ${initialSize}.")
      println(s"Expected result: histograms of size ${initialSize - stopIndex} to ${initialSize - startIndex} with step size $stepSize.")
      print("progress: ")
      backtrackedSpaced.foreach(h => print(s"${h.counts.leaves.length} -> "))
      println("done")
    } 
    backtrackedSpaced.toVector
  }

  def spacedHistToCRP(spacedHists: Iterable[Histogram], verbose: Boolean = false): CollatedHistogram[String] = {
    if (verbose) println(s"Merging ${spacedHists.size} histograms.")
    val spacedCRPs = spacedHists.zipWithIndex.map{ case (hist, i) => toCollatedHistogram[String](hist, s"$i") }
    if (verbose) print(s"progress: ${spacedCRPs.head.keySet.toVector.head}")
    val mergedCRPs = spacedCRPs.reduce[CollatedHistogram[String]]{ case (h1, h2) =>
      if (verbose) print(s", ${h2.keySet.toVector.head}")
      h1 collateNested h2
    }
    if (verbose) println
    mergedCRPs
  }

  /**
   * TODO: Memory issues on crpWithValMapBC, A real performance-hog, all parts
   */
  def getDelta(crp: CollatedHistogram[String], validationHist: Histogram, verbose: Boolean = false): Vector[(String, Double)] = {
    // Spark must be running
    val spark = getSpark   
    import spark.implicits._
    val numPartitions = spark.conf.get("spark.default.parallelism").toInt
    
    // keys are indices of histograms
    val crpKeys = crp.keySet.toVector.sortBy(_.toInt)
    
    if (verbose) println("Finding Scheffe Sets")
    
    // Find the Scheffe sets corresponding to all pairs of histograms
    val scheffeSets = {
      val crpDensVector = crp.densities.toIterable.toVector
      crp.keySet.toVector.flatMap(i => crpKeys.drop(i.toInt + 1).map((i,_))).map{ case (theta1, theta2) =>
        val keys = Set(theta1, theta2)
        (theta1, theta2) -> crpDensVector.filter{ case (_, densMap) => densMap(theta1)._1 > densMap(theta2)._1 }.map{case (node, _) => node}
      }
    }
    val scheffeDS = scheffeSets.map{ case (_, set) => set }.toDS.repartition(numPartitions).cache
    scheffeDS.count
    
    if (verbose) println("Broadcasting CRP with validation")
    
    /* TODO: Can skip string identifiers? let index represent identity instead. */
    // broadcast variable used in delta calculations.
    // Converted to arrays to avoid serialization issues (although the maps should be serializable...)
    type BCType = Broadcast[Array[(BigInt, Array[(String, Double)])]]
    val crpWithValMapBC: BCType = spark.sparkContext.broadcast(
      crp.collate(validationHist, "val").densities.toMap
        .map{ case (node, densMap) => 
          (node.lab, densMap.mapValues{ case (dens, vol) => dens * vol}.toArray)
        }.toArray
    )
    
    if (verbose) println("Calculating deviations from validation")
    
    // Wrrapped in function due to serialialization issues otherwise...
    /* TODO: Performance Hog! */
    val deviationsFromValidation = { crpWithValMapBC: BCType => 
      scheffeDS.map{ set => 
        val crpWithValMapLocal = crpWithValMapBC.value.map{ case (lab, arr) => (NodeLabel(lab), arr.toMap)}.toMap
        val crpAtSet = crpWithValMapLocal.filterKeys(key => set.contains(key))
        val validationMeasure = crpAtSet.values.map(_.apply("val")).sum
        crpAtSet.values.flatMap(_.toSeq).groupBy(_._1).map{ case (key, vals) => key -> math.abs(vals.unzip._2.sum - validationMeasure)}
      }.collect
    }.apply(crpWithValMapBC)
    
    // Return maximum deviation from validation measure for each histogram
    deviationsFromValidation
      .flatMap(_.filterKeys(_ != "val").toSeq)
      .groupBy(_._1)
      .map{ case (key, devs) => key -> devs.map(_._2).max }
      .toVector.sortBy(_._2)
  }

  def mdeStep(
    hist: Histogram, 
    validationData: RDD[(NodeLabel, Count)], 
    k: Int, 
    stopSize: Option[Int] = None, 
    verbose: Boolean = false
  ): (Histogram, Histogram) = {
    val spark = getSpark
    import spark.implicits._

    val stopIndex = hist.counts.leaves.length - stopSize.getOrElse(1)
    val stepSize = stopIndex / (k-1)
    if (verbose) println("--- Backtracking histogram ---")
    val backtrackedHist = spacedBacktrack(hist, 0, stopIndex, stepSize, verbose).reverse
    
    if (verbose) println("--- Merging CRPs ---")
    val crp = spacedHistToCRP(backtrackedHist, verbose)
    
    if (verbose) println("--- Computing validation data histogram ---")
    val maxCrpDepth = crp.densities.leaves.map(_.depth).max
    val crpLeafSet = crp.densities.leaves.toSet
    val crpLeafMap = crp.densities.copy(vals = Stream.continually(0).take(crp.densities.leaves.length).toVector)

    //val truncatedValData = validationDS.groupByKey{ case (node, _) => node.truncate(maxCrpDepth) }.mapGroups{ case (anc, nodesAndCounts) => (anc, nodesAndCounts.map{ case (_, count) => count}.sum) }
    
    /* TODO: [Performance] Only needs to be done once at the start, we never go deeper than the initial iteration  */
    val truncatedValData = validationData.map(t => (t._1.truncate(maxCrpDepth), t._2)).reduceByKey{(v1,v2) => v1 + v2}

    /*TODO: [Performance] Can see big improvements here by using SubtreePartitoning on Validation Data??? */
    val valHist = Histogram(
      hist.tree,
      truncatedValData.map(_._2).reduce(_+_),
      fromNodeLabelMap(
        { leafMap: LeafMap[_] =>
            truncatedValData.map(t => { (findSubtree(t._1, leafMap.truncation.leaves), t._2) }).reduceByKey((v1, v2) => v1 + v2)
        }.apply(crpLeafMap)
          .collect.toMap
      )
    )

   /* Old Dataset version (Creates non-correct histograms for some reason) */ 
   /*
    val valHist = Histogram(
      hist.tree,
      truncatedValData.map(_._2).reduce(_+_),
      fromNodeLabelMap(
        { leafMap: LeafMap[_] =>
          truncatedValData.toDS
            .groupByKey(node => leafMap.query((node._1 #:: node._1.ancestors).reverse)._1)
            .mapGroups{ case (node, nodesAndCounts) => (node, nodesAndCounts.map(_._2).sum) }
        }.apply(crpLeafMap)
        .collect.toMap
      )
    )
    */

    if (verbose) println("--- Computing histogram deviations from validation ---")
    /* Giga-hog of performance */
    val validationDeviations = getDelta(crp, valHist, verbose)
    
    val bestIndex = validationDeviations.head._1.toInt
    val bestHistogram = backtrackedHist(bestIndex)
    val largerThanBest = if (bestIndex < backtrackedHist.length - 1) backtrackedHist(bestIndex + 1) else backtrackedHist.last
    
    if (verbose && bestHistogram.counts.leaves.length == largerThanBest.counts.leaves.length)
      println("WARNING: best histogram is the largest, consider larger larger starting histogram!")

    (bestHistogram, largerThanBest)
  }

  /**
   * getMDE - Find the minimum distance estimate of an adaptive search within the path of increasingly more refined histograms, 
   *          with hist being the most refined histogram. 
   *
   * @param hist - The most refined histogram on the search path
   * @param validationData - The validation data to be used in finding a MDE in each search iteration.
   * @param k - The number of histograms to consider every iteration
   * @param verbose - Verbose printing of process
   *
   * @return The final non-normalized MDE from the whole adaptive search.
   */
  def getMDE(
    hist: Histogram, 
    validationData: RDD[(NodeLabel, Count)], 
    k: Int, 
    verbose: Boolean = false
  ): Histogram = {
    var stopSize = Option.empty[Int]
    if (verbose) println(s"----- Starting MDE search with step size $k and total leaf count ${hist.counts.leaves.length} -----")
    var bestAndLarger = mdeStep(hist, validationData, k, stopSize, verbose)
    var sizeDiff = bestAndLarger._2.counts.leaves.length - bestAndLarger._1.counts.leaves.length
    while (sizeDiff > k/2) {
      if (verbose) println(s"----- Current size difference: $sizeDiff -----")
      stopSize = Some(bestAndLarger._2.counts.leaves.length - 2 * sizeDiff)
      bestAndLarger = mdeStep(bestAndLarger._2, validationData, k, stopSize, verbose)
      sizeDiff = bestAndLarger._2.counts.leaves.length - bestAndLarger._1.counts.leaves.length
    }
    
    if (sizeDiff > 1) {
      if (verbose) println(s"----- Final step with size difference $sizeDiff -----")
      stopSize = Some(bestAndLarger._2.counts.leaves.length - 2 * sizeDiff)
      bestAndLarger = mdeStep(bestAndLarger._2, validationData, sizeDiff * 2 + 1, stopSize, verbose)
    }
    
    bestAndLarger._1
  }

}
