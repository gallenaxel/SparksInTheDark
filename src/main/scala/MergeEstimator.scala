package co.wiklund.disthist

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.mllib.linalg.Vectors

import Types._
import LeafMapFunctions._
import HistogramUtilityFunctions._
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.P

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
    var lastCount = tmpDS.count + 1
    while (currentDepth > 1 && tmpDS.count < lastCount) {
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
}