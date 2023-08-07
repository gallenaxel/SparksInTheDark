import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD

import math.{min, max, abs}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.LongAccumulator

import org.apache.log4j.{ Logger, Level }

import scala.math.BigInt
import java.math.BigInteger

import scala.reflect.io.Directory
import java.io.File

import co.wiklund.disthist._
import MDEFunctions._
import LeafMapFunctions._
import NodeLabelFunctions._
import SubtreePartitionerFunctions._
import HistogramFunctions._
import MergeEstimatorFunctions._
import org.apache.spark.mllib.linalg.{ Vector => MLVector }
import SpatialTreeFunctions._
import Types._

import org.scalatest.BeforeAndAfterAll
import org.scalatest.{ FlatSpec, Matchers }

class BlablaTests extends FlatSpec with Matchers with BeforeAndAfterAll {
  private val tn: Int => NodeLabel = NodeLabel(_)

  private var normalRDD: RDD[MLVector] = null
  private var valDS: Dataset[(NodeLabel, Count)] = null
  private val checkpointDir = "src/test/resources/dsCheckpoint"

  private val dfdim = 3
  private val dfnum = 5000
  private val valnum = 1000
  private val rootBox = Rectangle(
    (1 to dfdim).toVector.map(i => -10.0),
    (1 to dfdim).toVector.map(i => 10.0)
  )
  private val tree : WidestSplitTree = widestSideTreeRootedAt(rootBox)

  override protected def beforeAll: Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
    spark.conf.set("spark.default.parallelism", "6")

    normalRDD = normalVectorRDD(spark.sparkContext, dfnum, dfdim, 6, 1234)
    val valRDD = normalVectorRDD(spark.sparkContext, valnum, dfdim, 6, 4321)
    val labeledValRDD = labelAtDepth(tree, 40, valRDD)
    valDS = labeledToCountedDS(labeledValRDD)
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  override protected def afterAll: Unit = {
    val spark = getSpark
    spark.stop

    val chkPtDir = new Directory(new File(checkpointDir))
    chkPtDir.deleteRecursively
  }

  var backtrackedHistograms : Vector[Histogram] = null
  "getMDE" should "produce a correct histogram for the validation data in the rdd version and produce same delta as getMDEPrime" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 5
    val sizeExp = 5

    val numPartitions = 16
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-1

    val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
    val rawTestRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)

    var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
    var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectTest)

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = 1500 
    val kInMDE = 10
    println(finestResDepth)

    var countedTrain = quickToLabeled(tree, finestResDepth, rawTrainRDD)
    var countedTest = quickToLabeled(tree, finestResDepth, rawTestRDD)
        
    val partitioner = new SubtreePartitioner(2, countedTrain, 20) /* action 1 (collect) */
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 5 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit, true)
    println("merging done")

    val hist = Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap))
    var stopSize = Option.empty[Int]
    
    val stopIndex = 15000 
    val verbose = true
    if (verbose) println("--- Backtracking histogram ---")
    backtrackedHistograms = spacedBacktrack(hist, 0, stopIndex, stepSize, verbose).reverse
    
  //  if (verbose) println("--- Merging CRPs ---")
  //  val crp = spacedHistToCRP(backtrackedHistograms, verbose)
  //
  //  if (verbose) println("--- Computing validation data histogram ---")
  //  val maxCrpDepth = crp.densities.leaves.map(_.depth).max
  //  val crpLeafSet = crp.densities.leaves.toSet
  //  val crpLeafMap = crp.densities.copy(vals = Stream.continually(0).take(crp.densities.leaves.length).toVector)

  //  //val truncatedValData = validationDS.groupByKey{ case (node, _) => node.truncate(maxCrpDepth) }.mapGroups{ case (anc, nodesAndCounts) => (anc, nodesAndCounts.map{ case (_, count) => count}.sum) }
  //  
  //  /* TODO: [Performance] Only needs to be done once at the start, we never go deeper than the initial iteration  */
  //  val truncatedValData = countedTest.map(t => (t._1.truncate(maxCrpDepth), t._2)).reduceByKey{(v1,v2) => v1 + v2}

  //  /*TODO: [Performance] Can see big improvements here by using SubtreePartitoning on Validation Data??? */
  //  val valHist = Histogram(
  //    hist.tree,
  //    truncatedValData.map(_._2).reduce(_+_),
  //    fromNodeLabelMap(
  //      { leafMap: LeafMap[_] =>
  //          truncatedValData.map(t => { (findSubtree(t._1, leafMap.truncation.leaves), t._2) }).reduceByKey((v1, v2) => v1 + v2)
  //      }.apply(crpLeafMap)
  //        .collect.toMap
  //    )
  //  )
 
  //  if (verbose) println("--- Computing histogram deviations from validation ---")
  //  /* Giga-hog of performance */
  //  val k = backtrackedHistograms.length
  //  val validationDeviations = getDelta(crp, valHist, verbose)
  //  val bestIndex = validationDeviations.head._1.toInt
  //  val delta1 = validationDeviations.head._2
  //  val bestHistogram1 = backtrackedHistograms(bestIndex)

  //  val leaves = valHist.counts.truncation.leaves
  //  val counts = valHist.counts.vals
  //  assert(leaves.length == counts.length)
  //  assert(valHist.totalCount == (trainSize/2))
  //  for (i <- 0 until counts.length) {
  //    assert(counts(i) > 0)
  //    for (j <- (i+1) until counts.length) {
  //      assert(leftRightOrd.compare(leaves(i),leaves(j)) == -1)
  //      assert(!isAncestorOf(leaves(i), leaves(j)))
  //      assert(!isAncestorOf(leaves(j), leaves(i)))
  //    }
  //  }

  //  val validationData = countedTest.mapPartitions(_.toArray.sortBy(t => t._1)(leftRightOrd).toIterator)
  //  val validationCount = trainSize / 2
  //  var crpLeaves : Array[(NodeLabel, Array[Double])] = new Array(crp.densities.leaves.length)
  //  var crpValues : Array[Array[Double]] = Array.ofDim[Double](crp.densities.leaves.length, k)
  //  for (i <- 0 until crpLeaves.length) {
  //    for (j <- 0 until k) {
  //      crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
  //    }
  //    crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
  //  }

  //  if (verbose) println("--- Calculating Scheffe Set Integrals for histograms ---")

  //  var scheffeIntegrals : Array[Array[Array[Double]]] = Array.ofDim[Double](k,k,k)
  //  for (l <- 0 until crpLeaves.length) {
  //    val leaf = crpLeaves(l)
  //    val volume = crp.tree.volumeAt(leaf._1)
  //    for (i <- 0 until k) {
  //      for (j <- (i+1) until k) {
  //        if (leaf._2(i) > leaf._2(j))  {
  //          for (t <- 0 until k) {
  //            scheffeIntegrals(i)(j)(t) += volume * leaf._2(t)
  //          }
  //        } else if (leaf._2(i) < leaf._2(j)) {
  //          for (t <- 0 until k) {
  //            scheffeIntegrals(j)(i)(t) += volume * leaf._2(t)
  //          }
  //        }
  //      }
  //    }
  //  }

  //  if (verbose) println("--- Setting up count accumulators ---")
  //  var scheffeCountAccumulators : Array[Array[LongAccumulator]] = Array.ofDim[LongAccumulator](k,k)
  //  for (i <- 0 until k) {
  //    for (j <- (i+1) until k) {
  //      scheffeCountAccumulators(i)(j) = spark.sparkContext.longAccumulator(s"$i,$j")
  //      scheffeCountAccumulators(j)(i) = spark.sparkContext.longAccumulator(s"$j,$i")
  //    }
  //  }

  //  if (verbose) println("--- Calculating empirical measure over Scheffe sets ---")
  //  val mergedValidationData = validationData.mapPartitions(iter => scheffeSetsValidationCount(crpLeaves, iter, scheffeCountAccumulators)).cache

  //  mergedValidationData.count
  //  validationData.unpersist()

  //  var scheffeEmpiricals : Array[Array[Double]] = Array.ofDim[Double](k,k)
  //  for (i <- 0 until k) {
  //    for (j <- (i+1) until k) {
  //      scheffeEmpiricals(i)(j) = scheffeCountAccumulators(i)(j).value.toDouble / validationCount
  //      scheffeEmpiricals(j)(i) = scheffeCountAccumulators(j)(i).value.toDouble / validationCount
  //    }
  //  }

  //  scheffeEmpiricals.foreach(line => {
  //    for (i <- 0 until k) {
  //      print(line(i) + " ")
  //    }
  //    println("")
  //  })

  //  if (verbose) println("--- Finding the Minimum Delta  ---")
  //  var deltas : Array[Double] = new Array(k)
  //  for (i <- 0 until k) {
  //    for (j <- (i+1) until k) {
  //      for (t <- 0 until k) {
  //        var distance = abs(scheffeIntegrals(i)(j)(t) - scheffeEmpiricals(i)(j))
  //        deltas(t) = max(distance, deltas(t))
  //        distance = abs(scheffeIntegrals(j)(i)(t) - scheffeEmpiricals(j)(i))
  //        deltas(t) = max(distance, deltas(t))
  //      }
  //    }
  //  }

  //  var mdeIndex = 0 
  //  var delta2 = deltas(0)
  //  for (i <- 1 until k) {
  //    if (deltas(i) < deltas(mdeIndex)) {
  //      mdeIndex = i
  //      delta2 = deltas(i)
  //    }
  //  }
  //  println(delta1)
  //  println(delta2)
  //  assert(abs(delta1 - delta2) < 0.00001)
  }

  it should "produce a correct array representing the crp leaves" in {
    val crp = spacedHistToCRP(backtrackedHistograms, true)
    var crpLeaves : Array[(NodeLabel, Array[Double])] = new Array(crp.densities.leaves.length)
    val k = backtrackedHistograms.length
    var crpValues : Array[Array[Double]] = Array.ofDim[Double](crp.densities.leaves.length, k)
    for (i <- 0 until crpLeaves.length) {
      for (j <- 0 until k) {
        crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
      }
      crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
    }

    /* For each histogram leaf, determine leaves in crp, check density is correct */
    val crpTrunc = Truncation(crpLeaves.map(_._1).toVector)
    for (h <- 0 until backtrackedHistograms.length) {
      var next = 0
      for (i <- 0 until backtrackedHistograms(h).counts.truncation.leaves.length) {
        val leaf = backtrackedHistograms(h).counts.truncation.leaves(i) 
        val expected = (backtrackedHistograms(h).counts.vals(i).toDouble / backtrackedHistograms(h).totalCount) / backtrackedHistograms(h).tree.volumeAt(leaf)

        val ss = crpTrunc.subtree(leaf)
        /* most coarse histogram should traverse through exactly every crp leaf */
        if (h == 0) {
          assert(next == ss.lower)
        }

        next = ss.upper
        for (j <- ss.lower until ss.upper) {
          val actual = crpLeaves(j)._2(k-1-h)
          assert(abs(expected - actual) < 0.000000000001)
        }
        
      }
    }
  }

  it should "produce the correct result for a simple example" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val tree = WidestSplitTree(Rectangle(Vector(0.0), Vector(1.0)))
    val hist1 = Histogram(tree, 100, LeafMap(Truncation(Vector(NodeLabel(5), NodeLabel(6))), Vector(50, 50)))
    val hist2 = Histogram(tree, 100, LeafMap(Truncation(Vector(NodeLabel(10), NodeLabel(11), NodeLabel(12), NodeLabel(13))), Vector(40,10,40,10)))
    val crp = spacedHistToCRP(Vector(hist1, hist2))
    val k = 2

    var crpLeaves : Array[(NodeLabel, Array[Double])] = new Array(crp.densities.leaves.length)
    var crpValues : Array[Array[Double]] = Array.ofDim[Double](crp.densities.leaves.length, k)
    for (i <- 0 until crpLeaves.length) {
      for (j <- 0 until k) {
        crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
      }
      crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
    }

    val validationCount = 100
    val validationDataArray : Array[(NodeLabel,Count)] = Array(
      (NodeLabel(8), 1),
      (NodeLabel(9), 2),
      (NodeLabel(10), 37),
      (NodeLabel(11), 8),
      (NodeLabel(12), 38),
      (NodeLabel(13), 11),
      (NodeLabel(14), 1),
      (NodeLabel(15), 2),
    )

    var validationData = spark.sparkContext.parallelize(validationDataArray, 4)
    var scheffeIntegrals : Array[Array[Array[Double]]] = Array.ofDim[Double](k,k,k)
    for (l <- 0 until crpLeaves.length) {
      val leaf = crpLeaves(l)
      val volume = crp.tree.volumeAt(leaf._1)
      for (i <- 0 until k) {
        for (j <- (i+1) until k) {
          if (leaf._2(i) > leaf._2(j))  {
            for (t <- 0 until k) {
              scheffeIntegrals(i)(j)(t) += volume * leaf._2(t)
            }
          } else if (leaf._2(i) < leaf._2(j)) {
            for (t <- 0 until k) {
              scheffeIntegrals(j)(i)(t) += volume * leaf._2(t)
            }
          }
        }
      }
    }

    var scheffeCountAccumulators : Array[Array[LongAccumulator]] = Array.ofDim[LongAccumulator](k,k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        scheffeCountAccumulators(i)(j) = spark.sparkContext.longAccumulator(s"$i,$j")
        scheffeCountAccumulators(j)(i) = spark.sparkContext.longAccumulator(s"$j,$i")
      }
    }

    val mergedValidationData = validationData.mapPartitions(iter => scheffeSetsValidationCount(crpLeaves, iter, scheffeCountAccumulators)).cache

    mergedValidationData.count
    validationData.unpersist()

    var scheffeEmpiricals : Array[Array[Double]] = Array.ofDim[Double](k,k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        scheffeEmpiricals(i)(j) = scheffeCountAccumulators(i)(j).value.toDouble / validationCount
        scheffeEmpiricals(j)(i) = scheffeCountAccumulators(j)(i).value.toDouble / validationCount
      }
    }

    scheffeEmpiricals.foreach(line => {
      for (i <- 0 until k) {
        print(line(i) + " ")
      }
      println("")
    })

    assert(abs(scheffeEmpiricals(0)(1) - 0.75) < 0.000000001)
    assert(abs(scheffeEmpiricals(1)(0) - 0.19) < 0.000000001)

    assert(abs(scheffeIntegrals(0)(1)(0) - 0.8) < 0.000000001)
    assert(abs(scheffeIntegrals(0)(1)(1) - 0.5) < 0.000000001)
    assert(abs(scheffeIntegrals(1)(0)(0) - 0.2) < 0.000000001)
    assert(abs(scheffeIntegrals(1)(0)(1) - 0.5) < 0.000000001)

    var deltas : Array[Double] = new Array(k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        for (t <- 0 until k) {
          var distance = abs(scheffeIntegrals(i)(j)(t) - scheffeEmpiricals(i)(j))
          deltas(t) = max(distance, deltas(t))
          distance = abs(scheffeIntegrals(j)(i)(t) - scheffeEmpiricals(j)(i))
          deltas(t) = max(distance, deltas(t))
        }
      }
    }

    var mdeIndex = 0 
    for (i <- 1 until k) {
      if (deltas(i) < deltas(mdeIndex)) {
        mdeIndex = i
      }
    }

    assert(abs(deltas(0) - 0.05) < 0.000000001)
    assert(abs(deltas(1) - 0.31) < 0.000000001)
    assert(abs(deltas(mdeIndex) - 0.05) < 0.000000001)

    //val maxCrpDepth = crp.densities.leaves.map(_.depth).max
    //val crpLeafSet = crp.densities.leaves.toSet
    //val crpLeafMap = crp.densities.copy(vals = Stream.continually(0).take(crp.densities.leaves.length).toVector)

    //val truncatedValData = validationData.map(t => (t._1.truncate(maxCrpDepth), t._2)).reduceByKey{(v1,v2) => v1 + v2}
    //val valHist = Histogram(
    //  tree,
    //  truncatedValData.map(_._2).reduce(_+_),
    //  fromNodeLabelMap(
    //    { leafMap: LeafMap[_] =>
    //        truncatedValData.map(t => { (findSubtree(t._1, leafMap.truncation.leaves), t._2) }).reduceByKey((v1, v2) => v1 + v2)
    //    }.apply(crpLeafMap)
    //      .collect.toMap
    //  )
    //)

    //val validationDeviations = getDelta(crp, valHist)

    //val bestDelta = validationDeviations.head._2
    //assert(abs(bestDelta - 0.05) < 0.000000001)
  }
}

class MDETests extends FlatSpec with Matchers with BeforeAndAfterAll {
  private val tn: Int => NodeLabel = NodeLabel(_)

  private var normalRDD: RDD[MLVector] = null
  private var valDS: Dataset[(NodeLabel, Count)] = null
  private val checkpointDir = "src/test/resources/dsCheckpoint"

  private val dfdim = 3
  private val dfnum = 5000
  private val valnum = 1000
  private val rootBox = Rectangle(
    (1 to dfdim).toVector.map(i => -10.0),
    (1 to dfdim).toVector.map(i => 10.0)
  )
  private val tree : WidestSplitTree = widestSideTreeRootedAt(rootBox)

  override protected def beforeAll: Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
    spark.conf.set("spark.default.parallelism", "6")

    normalRDD = normalVectorRDD(spark.sparkContext, dfnum, dfdim, 6, 1234)
    val valRDD = normalVectorRDD(spark.sparkContext, valnum, dfdim, 6, 4321)
    val labeledValRDD = labelAtDepth(tree, 40, valRDD)
    valDS = labeledToCountedDS(labeledValRDD)
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  override protected def afterAll: Unit = {
    val spark = getSpark
    spark.stop

    val chkPtDir = new Directory(new File(checkpointDir))
    chkPtDir.deleteRecursively
  }

  "labelAtDepth" should "give correct labels" in {
    val spark = getSpark
    import spark.implicits._
    val testDS = Vector(
      Array(1.0, 1.0, 1.0),
      Array(-1.0, 1.0, 1.0),
      Array(1.0, -1.0, 1.0),
      Array(-1.0, -1.0, 1.0)
    ).toDS

    val labeledDS = labelAtDepth(tree, 2, testDS)
    val labels = labeledDS.collect.map(_._1).toVector.sorted(leftRightOrd)
    val expectedLabels = Vector(4,5,6,7) map tn
    labels shouldEqual expectedLabels
  }

  "labeledToCountedDS" should "preserve count" in {
    val spark = getSpark
    import spark.implicits._
    val labeledRDD = labelAtDepth(tree, 4, normalRDD)
    val countedDS = labeledToCountedDS(labeledRDD)
    val totalCount = countedDS.map(_._2).reduce(_+_)
    totalCount shouldEqual dfnum
  }

  "quickToLabeled" should "give correct labels" in {
    val spark = getSpark
    import spark.implicits._
    val testRDD = spark.sparkContext.parallelize(Vector(
      Vectors.dense(1.0, 1.0, 1.0),
      Vectors.dense(-1.0, 1.0, 1.0),
      Vectors.dense(1.0, -1.0, 1.0),
      Vectors.dense(-1.0, -1.0, 1.0)
    ))

    val labeled = quickToLabeled(tree, 2, testRDD)
    val labels = labeled.collect.map(_._1).toVector.sorted(leftRightOrd)
    val expectedLabels = Vector(4,5,6,7) map tn
    labels shouldEqual expectedLabels
  }

  it should "preserve count" in {
    val spark = getSpark
    import spark.implicits._
    val labeled = quickToLabeled(tree, 4, normalRDD)
    val totalCount = labeled.collect.map(_._2).reduce(_+_)
    assert(totalCount == dfnum)
  }

  it should "generate the same labels and counts as older versions" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 10
    val sizeExp = 4

    val numPartitions = 4
    
    val trainSize = math.pow(10, sizeExp).toLong

    val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567).cache

    var rootBox = RectangleFunctions.boundingBox(rawTrainRDD)

    val tree = widestSideTreeRootedAt(rootBox)

    println("Starting Regression Test on new labeling routine")
    for (depth <- 1 until 50) {
      println("Depth: " + depth)
      var tmp1 = quickToLabeled(tree, depth, rawTrainRDD).collect.sortBy(_._1)(leftRightOrd)
      val labeledRDD = labelAtDepth(tree, depth, rawTrainRDD)
      val tmp2 = labeledToCountedDS(labeledRDD).collect.sortBy(_._1)(leftRightOrd)
      assert(tmp1.length == tmp2.length)
      for (i <- 0 until tmp1.length) {
        assert(tmp1(i)._1 == tmp2(i)._1)
        assert(tmp1(i)._2 == tmp2(i)._2)
      }
    }
  }

  "mergeLeaves" should "respect count limit" in {
    val spark = getSpark
    import spark.implicits._
    val labeledRDD = labelAtDepth(tree, 15, normalRDD)
    val countedDS = labeledToCountedDS(labeledRDD)

    val countLimit = countedDS.map(_._2).collect.max
    val stepSize = 4
    val mergedDS = mergeLeaves(tree, countedDS, countLimit, stepSize, checkpointDir + "/merged")
    val numLeavesAboveLimit = mergedDS.filter(_._2 > countLimit).count
    numLeavesAboveLimit shouldEqual 0
  }

    "getMDEPrime" should "give a correct histogram" in {
    val spark = getSpark
    import spark.implicits._
    val mergedHist = collectHistogram(tree, spark.read.parquet(checkpointDir + "/merged").as[(NodeLabel, Count)])
    val mdeHist = getMDEPrime(mergedHist, valDS.rdd, 5, true)

    mdeHist.counts.vals.sum shouldEqual dfnum
  }


  it should "produce a correct histogram for the validation data in the rdd version and produce same delta as getMDE" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 5
    val sizeExp = 5

    val numPartitions = 16
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-1

    val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
    val rawTestRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)

    var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
    var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectTest)

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = 1500 
    val kInMDE = 10
    println(finestResDepth)

    var countedTrain = quickToLabeled(tree, finestResDepth, rawTrainRDD)
    var countedTest = quickToLabeled(tree, finestResDepth, rawTestRDD)
        
    val partitioner = new SubtreePartitioner(2, countedTrain, 20) /* action 1 (collect) */
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 5 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit, true)
    println("merging done")

    val hist = Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap))
    var stopSize = Option.empty[Int]
    
    val stopIndex = 15000 
    val verbose = true
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
    val truncatedValData = countedTest.map(t => (t._1.truncate(maxCrpDepth), t._2)).reduceByKey{(v1,v2) => v1 + v2}

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
    /* Giga-hog of performance */
    val k = backtrackedHist.length
    val validationDeviations = getDelta(crp, valHist, verbose)
    val bestIndex = validationDeviations.head._1.toInt
    val delta1 = validationDeviations.head._2
    val bestHistogram1 = backtrackedHist(bestIndex)

    val leaves = valHist.counts.truncation.leaves
    val counts = valHist.counts.vals
    assert(leaves.length == counts.length)
    assert(valHist.totalCount == (trainSize/2))
    for (i <- 0 until counts.length) {
      assert(counts(i) > 0)
      for (j <- (i+1) until counts.length) {
        assert(leftRightOrd.compare(leaves(i),leaves(j)) == -1)
        assert(!isAncestorOf(leaves(i), leaves(j)))
        assert(!isAncestorOf(leaves(j), leaves(i)))
      }
    }

    val validationData = countedTest
    val validationCount = trainSize / 2
    var crpLeaves : Array[(NodeLabel, Array[Double])] = new Array(crp.densities.leaves.length)
    var crpValues : Array[Array[Double]] = Array.ofDim[Double](crp.densities.leaves.length, k)
    for (i <- 0 until crpLeaves.length) {
      for (j <- 0 until k) {
        crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
      }
      crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
    }

    if (verbose) println("--- Calculating Scheffe Set Integrals for histograms ---")

    var scheffeIntegrals : Array[Array[Array[Double]]] = Array.ofDim[Double](k,k,k)
    for (l <- 0 until crpLeaves.length) {
      val leaf = crpLeaves(l)
      val volume = crp.tree.volumeAt(leaf._1)
      for (i <- 0 until k) {
        for (j <- (i+1) until k) {
          if (leaf._2(i) > leaf._2(j))  {
            for (t <- 0 until k) {
              scheffeIntegrals(i)(j)(t) += volume * leaf._2(t)
            }
          } else if (leaf._2(i) < leaf._2(j)) {
            for (t <- 0 until k) {
              scheffeIntegrals(j)(i)(t) += volume * leaf._2(t)
            }
          }
        }
      }
    }

    if (verbose) println("--- Setting up count accumulators ---")
    var scheffeCountAccumulators : Array[Array[LongAccumulator]] = Array.ofDim[LongAccumulator](k,k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        scheffeCountAccumulators(i)(j) = spark.sparkContext.longAccumulator(s"$i,$j")
        scheffeCountAccumulators(j)(i) = spark.sparkContext.longAccumulator(s"$j,$i")
      }
    }

    if (verbose) println("--- Calculating empirical measure over Scheffe sets ---")
    val mergedValidationData = validationData.mapPartitions(iter => scheffeSetsValidationCount(crpLeaves, iter, scheffeCountAccumulators)).cache

    mergedValidationData.count
    validationData.unpersist()

    var scheffeEmpiricals : Array[Array[Double]] = Array.ofDim[Double](k,k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        scheffeEmpiricals(i)(j) = scheffeCountAccumulators(i)(j).value.toDouble / validationCount
        scheffeEmpiricals(j)(i) = scheffeCountAccumulators(j)(i).value.toDouble / validationCount
      }
    }

    scheffeEmpiricals.foreach(line => {
      for (i <- 0 until k) {
        print(line(i) + " ")
      }
      println("")
    })

    if (verbose) println("--- Finding the Minimum Delta  ---")
    var deltas : Array[Double] = new Array(k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        for (t <- 0 until k) {
          var distance = abs(scheffeIntegrals(i)(j)(t) - scheffeEmpiricals(i)(j))
          deltas(t) = max(distance, deltas(t))
          distance = abs(scheffeIntegrals(j)(i)(t) - scheffeEmpiricals(j)(i))
          deltas(t) = max(distance, deltas(t))
        }
      }
    }

    var mdeIndex = 0 
    var delta2 = deltas(0)
    for (i <- 1 until k) {
      if (deltas(i) < deltas(mdeIndex)) {
        mdeIndex = i
        delta2 = deltas(i)
      }
    }

    assert(abs(delta1 - delta2) < 0.000000001)
  }

  "Histogram.density" should "produce 0 values at 0-probability regions" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 10
    val sizeExp = 5

    val numPartitions = 16
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1 

    val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
    val rawTestRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)

    var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
    var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectTest)

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = 1500 
    val kInMDE = 10
    println(finestResDepth)

    var countedTrain = quickToLabeled(tree, finestResDepth, rawTrainRDD)
    var countedTest = quickToLabeled(tree, finestResDepth, rawTestRDD)
        
    val partitioner = new SubtreePartitioner(2, countedTrain, 20) /* action 1 (collect) */
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 10
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit, true)

    val hist = Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap))

    val outsidePoint1 = normalVectorRDD(spark.sparkContext, 1, dimensions, numPartitions, 103032)
      .map(x => Vectors.dense(x.toArray.map(x => x + 100.0))).collect
    assert(hist.density(outsidePoint1(0)) == 0.0)
    
    val last = subtreeRDD.collect.last._1.lab
    var leftTurn = false
    for (i <- 0 until 70) {
      leftTurn = leftTurn || !last.testBit(i)
    }
    assert(leftTurn)
  }

  "getMDE" should "give same results as old implementation" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    for (dimensions <- Seq(1,2,5)) {
      val sizeExp = 6
      val numPartitions = 16
      
      val trainSize = math.pow(10, sizeExp).toLong
      val finestResSideLength = 0.00001

      val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
      val rawTestRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)

      var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
      var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
      val rootBox = RectangleFunctions.hull(rectTrain, rectTest)

      val tree = widestSideTreeRootedAt(rootBox)
      val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
      val stepSize = 1500 
      val kInMDE = 10
      println(finestResDepth)

      var countedTrain = quickToLabeled(tree, finestResDepth, rawTrainRDD)
      var countedTest = quickToLabeled(tree, finestResDepth, rawTestRDD)
          
      val partitioner = new SubtreePartitioner(2, countedTrain, 20) /* action 1 (collect) */
      val depthLimit = partitioner.maxSubtreeDepth
      val countLimit = 100
      val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
      val finestHistogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit, true)
    
      val newH = getMDE(finestHistogram, countedTest, trainSize / 2, kInMDE, true)
      val oldH = getMDEPrime(finestHistogram, countedTest, kInMDE, true)

      assert(newH == oldH)
    }
  }
}
