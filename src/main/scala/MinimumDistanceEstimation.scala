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
    
    if (verbose) println("--- Computing histogram deviations from validation ---")
    val validationDeviations = getDelta(crp, valHist, verbose)
    
    val bestIndex = validationDeviations.head._1.toInt
    val bestHistogram = backtrackedHist(bestIndex)
    val largerThanBest = if (bestIndex < backtrackedHist.length - 1) backtrackedHist(bestIndex + 1) else backtrackedHist.last
    
    if (verbose && bestHistogram.counts.leaves.length == largerThanBest.counts.leaves.length)
      println("WARNING: best histogram is the largest, consider larger larger starting histogram!")

    (bestHistogram, largerThanBest)
  }

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
