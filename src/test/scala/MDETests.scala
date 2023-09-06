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

  var backtrackedHistograms : Vector[Histogram] = null
  "getMDE" should "produce a correct array representing the crp leaves" in {
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

  it should "produce the correct validationDataRDD for every iteration: " in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 2
    val sizeExp = 6

    val numPartitions = 16
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-5

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
    val countLimit = 200 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit, true)
    val verbose = true 
    var stopSize = Option.empty[Int]

    def getValHist (hist: Histogram, validationData: RDD[(NodeLabel, Count)], k: Int, stopSize: Option[Int] = None) : Vector[(NodeLabel,Count)] = {
      val stopIndex = hist.counts.leaves.length - stopSize.getOrElse(1)
      val stepSize = stopIndex / (k-1)
      val backtrackedHist = spacedBacktrack(hist, 0, stopIndex, stepSize, verbose).reverse
      val h = backtrackedHist.length
      
      val crp = spacedHistToCRP(backtrackedHist, verbose)
      val maxCrpDepth = crp.densities.leaves.map(_.depth).max
      val crpLeafSet = crp.densities.leaves.toSet
      val crpLeafMap = crp.densities.copy(vals = Stream.continually(0).take(crp.densities.leaves.length).toVector)
      val lls = crp.densities.truncation.leaves

      validationData
        .map(t => (t._1.truncate(maxCrpDepth), t._2))
        .map(l => { 
          val t = findSubtree(l._1, lls)
          if (isAncestorOf(t, l._1) || t == l._1) {
            (t, l._2) 
          } else {
            l
          }
        })
        .reduceByKey((v1, v2) => v1 + v2)
        .collect
        .sortBy(_._1)
        .toVector
    }

    def assertLeaves(v1 : Vector[(NodeLabel, Count)], v2 : Vector[(NodeLabel,Count)]) = {
      for (i <- 0 until v1.length) {
        if (v1(i)._1 != v2(i)._1 || v1(i)._2 != v2(i)._2) {
          println("---ERROR---")
          println("INDEX: " + i)
          println("LENGHTS: " + v1.length + ", " + v2.length)
          println("LABELS: " + v1(i)._1 + ", " + v2(i)._1)
          println("COUNTS: " + v1(i)._2 + ", " + v2(i)._2)
          println("COUNT SUMS: " + v1.map(_._2).reduce(_+_) + ", " + v2.map(_._2).reduce(_+_))
          println("AFTER LABELS: " + v1(i+1)._1 + ", " + v2(i+1)._1)
          println("AFTER COUNTS: " + v1(i+1)._2 + ", " + v2(i+1)._2)
        }
        assert(v1(i)._1 == v2(i)._1)
        assert(v1(i)._2 == v2(i)._2)
      }
      assert(v1.length == v2.length)
    }

    val k = kInMDE
    var result : (Histogram, Histogram, RDD[(NodeLabel,Count)]) =
      mdeStep(merged, countedTest.mapPartitions(_.toArray.sortBy(t => t._1)(leftRightOrd).toIterator), trainSize/2, k, stopSize, 4, verbose)
    var best = result._1
    var largest = result._2
    var mergedValidationData = result._3
    var correct = getValHist(merged, countedTest, k, stopSize)
    var sizeDiff = largest.counts.leaves.length - best.counts.leaves.length
    assertLeaves(mergedValidationData.reduceByKey(_+_).collect.sortBy(_._1).toVector, correct)

    while (sizeDiff > k/2) {
      if (verbose) println(s"----- Current size difference: $sizeDiff -----")
      stopSize = Some(largest.counts.leaves.length - 2 * sizeDiff)
      result = mdeStep(largest, mergedValidationData, trainSize/2, k, stopSize, 4, verbose)
      correct = getValHist(largest, mergedValidationData, k, stopSize)
      best = result._1
      largest = result._2
      mergedValidationData = result._3
      assertLeaves(mergedValidationData.reduceByKey(_+_).collect.sortBy(_._1).toVector, correct)
      sizeDiff = largest.counts.leaves.length - best.counts.leaves.length
    }
    
    if (sizeDiff > 1) {
      if (verbose) println(s"----- Final step with size difference $sizeDiff -----")
      stopSize = Some(largest.counts.leaves.length - 2 * sizeDiff)
      result = mdeStep(largest, mergedValidationData, trainSize/2, sizeDiff * 2 + 1, stopSize, 4, verbose)
      correct = getValHist(largest, mergedValidationData, sizeDiff * 2 + 1, stopSize)
      best = result._1
      largest = result._2
      mergedValidationData = result._3
      assertLeaves(mergedValidationData.reduceByKey(_+_).collect.sortBy(_._1).toVector, correct)
    }
  }

  it should "produce the correct mergedValidationData when leaves are to the left, right and inbetween the CRP Leaves: " in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val tree = WidestSplitTree(Rectangle(Vector(0.0), Vector(1.0)))
    val hist = Histogram(tree, 100, LeafMap(Truncation(Vector(NodeLabel(20), NodeLabel(24))), Vector(49, 51)))
    val stopSize = Option(1)
    val verbose = false
    val k = 2

    val validationData1 : RDD[(NodeLabel, Count)] = spark.sparkContext.parallelize(Vector(
        (NodeLabel(32), 5),
        (NodeLabel(33), 5),
        (NodeLabel(17), 5),
        (NodeLabel(18), 5),
        (NodeLabel(76), 5),
        (NodeLabel(77), 5),
        (NodeLabel(39), 5),
        (NodeLabel(20), 5),
        (NodeLabel(24), 5),
      ), 4)
    val validationCount1 = 45
    var result1 : (Histogram, Histogram, RDD[(NodeLabel,Count)]) = mdeStep(hist, validationData1.mapPartitions(_.toArray.sortBy(t => t._1)(leftRightOrd).toIterator), validationCount1, k, stopSize, 4, verbose)
    var mergedValidationData1 = result1._3.reduceByKey(_+_).collect.sortBy(_._1)
    assert(mergedValidationData1(0)._1 == NodeLabel(16) && mergedValidationData1(0)._2 == 10)
    assert(mergedValidationData1(1)._1 == NodeLabel(17) && mergedValidationData1(1)._2 == 5)
    assert(mergedValidationData1(2)._1 == NodeLabel(18) && mergedValidationData1(2)._2 == 5)
    assert(mergedValidationData1(3)._1 == NodeLabel(19) && mergedValidationData1(3)._2 == 15)
    assert(mergedValidationData1(4)._1 == NodeLabel(20) && mergedValidationData1(4)._2 == 5)
    assert(mergedValidationData1(5)._1 == NodeLabel(24) && mergedValidationData1(5)._2 == 5)

    val validationData2 : RDD[(NodeLabel, Count)] = spark.sparkContext.parallelize(Vector(
        (NodeLabel(20), 5),
        (NodeLabel(24), 5),
        (NodeLabel(100), 5),
        (NodeLabel(101), 10),
        (NodeLabel(51), 10),
        (NodeLabel(26), 5),
        (NodeLabel(27), 5),
        (NodeLabel(112), 5),
        (NodeLabel(113), 10),
        (NodeLabel(57), 10),
        (NodeLabel(29), 5),
        (NodeLabel(30), 5),
        (NodeLabel(124), 5),
        (NodeLabel(125), 10),
        (NodeLabel(63), 10),
      ), 8)
    val validationCount2 = 105
    var result2 : (Histogram, Histogram, RDD[(NodeLabel,Count)]) = mdeStep(hist, validationData2.mapPartitions(_.toArray.sortBy(t => t._1)(leftRightOrd).toIterator), validationCount2, k, stopSize, 4, verbose)
    var mergedValidationData2 = result2._3.reduceByKey(_+_).collect.sortBy(_._1)
    assert(mergedValidationData2(0)._1 == NodeLabel(20) && mergedValidationData2(0)._2 == 5)
    assert(mergedValidationData2(1)._1 == NodeLabel(24) && mergedValidationData2(1)._2 == 5)
    assert(mergedValidationData2(2)._1 == NodeLabel(25) && mergedValidationData2(2)._2 == 25)
    assert(mergedValidationData2(3)._1 == NodeLabel(26) && mergedValidationData2(3)._2 == 5)
    assert(mergedValidationData2(4)._1 == NodeLabel(27) && mergedValidationData2(4)._2 == 5)
    assert(mergedValidationData2(5)._1 == NodeLabel(28) && mergedValidationData2(5)._2 == 25)
    assert(mergedValidationData2(6)._1 == NodeLabel(29) && mergedValidationData2(6)._2 == 5)
    assert(mergedValidationData2(7)._1 == NodeLabel(30) && mergedValidationData2(7)._2 == 5)
    assert(mergedValidationData2(8)._1 == NodeLabel(31) && mergedValidationData2(8)._2 == 25)

    val validationData3 : RDD[(NodeLabel, Count)] = spark.sparkContext.parallelize(Vector(
        (NodeLabel(20), 5),
        (NodeLabel(43), 10),
        (NodeLabel(88), 5),
        (NodeLabel(89), 10),
        (NodeLabel(45), 10),
        (NodeLabel(92), 5),
        (NodeLabel(93), 10),
        (NodeLabel(47), 10),
        (NodeLabel(24), 5),
      ), 2)
    val validationCount3  = 85
    var result3 : (Histogram, Histogram, RDD[(NodeLabel,Count)]) = mdeStep(hist, validationData3.mapPartitions(_.toArray.sortBy(t => t._1)(leftRightOrd).toIterator), validationCount3, k, stopSize, 4, verbose)
    var mergedValidationData3 = result3._3.reduceByKey(_+_).collect.sortBy(_._1)
    assert(mergedValidationData3(0)._1 == NodeLabel(20) && mergedValidationData3(0)._2 == 5)
    assert(mergedValidationData3(2)._1 == NodeLabel(22) && mergedValidationData3(2)._2 == 25)
    assert(mergedValidationData3(3)._1 == NodeLabel(23) && mergedValidationData3(3)._2 == 25)
    assert(mergedValidationData3(4)._1 == NodeLabel(24) && mergedValidationData3(4)._2 == 5)
  }

  it should "produce the correct mergedValidationData in a simple example: " in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val validationCount = 32
    val validationDataArray : Array[(NodeLabel, Count)] = Array(
      (NodeLabel(32), 1), 
      (NodeLabel(33), 1),
      (NodeLabel(34), 1),
      (NodeLabel(35), 1),
      (NodeLabel(36), 1),
      (NodeLabel(37), 1),
      (NodeLabel(38), 1),
      (NodeLabel(39), 1),
      (NodeLabel(40), 1),
      (NodeLabel(41), 1),
      (NodeLabel(42), 1),
      (NodeLabel(43), 1),
      (NodeLabel(44), 1),
      (NodeLabel(45), 1),
      (NodeLabel(46), 1),
      (NodeLabel(47), 1),
      (NodeLabel(48), 1),
      (NodeLabel(49), 1),
      (NodeLabel(50), 1),
      (NodeLabel(51), 1),
      (NodeLabel(52), 1),
      (NodeLabel(53), 1),
      (NodeLabel(54), 1),
      (NodeLabel(55), 1),
      (NodeLabel(56), 1),
      (NodeLabel(57), 1),
      (NodeLabel(58), 1),
      (NodeLabel(59), 1),
      (NodeLabel(60), 1),
      (NodeLabel(61), 1),
      (NodeLabel(62), 1),
      (NodeLabel(63), 1),
    )
    val validationData : RDD[(NodeLabel,Count)] = spark.sparkContext.parallelize(validationDataArray, 4)


    val tree = WidestSplitTree(Rectangle(Vector(0.0), Vector(1.0)))
    val hist1 = Histogram(tree, 100, LeafMap(Truncation(Vector(NodeLabel(10), NodeLabel(11), NodeLabel(14), NodeLabel(30))), Vector(23, 24, 21, 32)))
    val hist2 = Histogram(tree, 100, LeafMap(Truncation(Vector(NodeLabel(5), NodeLabel(14), NodeLabel(15))), Vector(47, 21, 32)))
    val hist3 = Histogram(tree, 100, LeafMap(Truncation(Vector(NodeLabel(2), NodeLabel(3))), Vector(47, 53)))
    val verbose = true 
    val k = 2

    var crp = spacedHistToCRP(Vector(hist1, hist2))
    var crpLeaves : Array[(NodeLabel, Array[Double])] = new Array(crp.densities.leaves.length)
    var crpValues : Array[Array[Double]] = Array.ofDim[Double](crp.densities.leaves.length, k)
    var crpMaxDepth = 0

    for (i <- 0 until crpLeaves.length) {
      for (j <- 0 until k) {
        crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
      }
      crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
      crpMaxDepth = max(crpMaxDepth, crp.densities.truncation.leaves(i).depth)
    }

    assert(crpMaxDepth == 4)

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
    
    val mergedValidationData1 = validationData.mapPartitions(iter => scheffeSetsValidationCount(crpLeaves, iter, crpMaxDepth, scheffeCountAccumulators)).cache

    mergedValidationData1.count
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

    val arr1 = mergedValidationData1.reduceByKey(_+_).collect.sortBy(_._1)
    assert(arr1(0)._1 == NodeLabel(16) && arr1(0)._2 == 2)
    assert(arr1(1)._1 == NodeLabel(17) && arr1(1)._2 == 2)
    assert(arr1(2)._1 == NodeLabel(18) && arr1(2)._2 == 2)
    assert(arr1(3)._1 == NodeLabel(19) && arr1(3)._2 == 2)
    assert(arr1(4)._1 == NodeLabel(10) && arr1(4)._2 == 4)
    assert(arr1(5)._1 == NodeLabel(11) && arr1(5)._2 == 4)
    assert(arr1(6)._1 == NodeLabel(24) && arr1(6)._2 == 2)
    assert(arr1(7)._1 == NodeLabel(25) && arr1(7)._2 == 2)
    assert(arr1(8)._1 == NodeLabel(26) && arr1(8)._2 == 2)
    assert(arr1(9)._1 == NodeLabel(27) && arr1(9)._2 == 2)
    assert(arr1(10)._1 == NodeLabel(14) && arr1(10)._2 == 4)
    assert(arr1(11)._1 == NodeLabel(30) && arr1(11)._2 == 2)
    assert(arr1(12)._1 == NodeLabel(31) && arr1(12)._2 == 2)

    assert(abs(scheffeEmpiricals(0)(1) - 6.0/32.0) < 0.000000001)
    assert(abs(scheffeEmpiricals(1)(0) - 6.0/32.0) < 0.000000001)

    assert(abs(scheffeIntegrals(0)(1)(0) - 0.395) < 0.000000001)
    assert(abs(scheffeIntegrals(0)(1)(1) - 0.23) < 0.000000001)
    assert(abs(scheffeIntegrals(1)(0)(0) - 0.395) < 0.000000001)
    assert(abs(scheffeIntegrals(1)(0)(1) - 0.56) < 0.000000001)

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

    assert(abs(deltas(0) - (0.395-6.0/32.0)) < 0.000000001)
    assert(abs(deltas(1) - (0.56-6.0/32.0)) < 0.000000001)
    assert(abs(deltas(mdeIndex) - (0.395-6.0/32.0)) < 0.000000001)


    crp = spacedHistToCRP(Vector(hist2, hist3))
    crpLeaves = new Array(crp.densities.leaves.length)
    crpValues = Array.ofDim[Double](crp.densities.leaves.length, k)
    crpMaxDepth = 0

    for (i <- 0 until crpLeaves.length) {
      for (j <- 0 until k) {
        crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
      }
      crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
      crpMaxDepth = max(crpMaxDepth, crp.densities.truncation.leaves(i).depth)
    }

    scheffeIntegrals = Array.ofDim[Double](k,k,k)
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

    scheffeCountAccumulators = Array.ofDim[LongAccumulator](k,k)
    for (i <- 0 until k) {
      for (j <- (i+1) until k) {
        scheffeCountAccumulators(i)(j) = spark.sparkContext.longAccumulator(s"$i,$j")
        scheffeCountAccumulators(j)(i) = spark.sparkContext.longAccumulator(s"$j,$i")
      }
    }
    
    val mergedValidationData2 = mergedValidationData1.mapPartitions(iter => scheffeSetsValidationCount(crpLeaves, iter, crpMaxDepth, scheffeCountAccumulators)).cache

    mergedValidationData2.count
    mergedValidationData1.unpersist()

    val arr2 = mergedValidationData2.reduceByKey(_+_).collect.sortBy(_._1)
    assert(arr2(0)._1 == NodeLabel(4) && arr2(0)._2 == 8)
    assert(arr2(1)._1 == NodeLabel(5) && arr2(1)._2 == 8)
    assert(arr2(2)._1 == NodeLabel(6) && arr2(2)._2 == 8)
    assert(arr2(3)._1 == NodeLabel(14) && arr2(3)._2 == 4)
    assert(arr2(4)._1 == NodeLabel(15) && arr2(4)._2 == 4)

    scheffeEmpiricals = Array.ofDim[Double](k,k)
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

    assert(abs(scheffeEmpiricals(0)(1) - 0.5) < 0.000000001)
    assert(abs(scheffeEmpiricals(1)(0) - 0.5) < 0.000000001)

    assert(abs(scheffeIntegrals(0)(1)(0) - 0.5) < 0.000000001)
    assert(abs(scheffeIntegrals(0)(1)(1) - 0.0) < 0.000000001)
    assert(abs(scheffeIntegrals(1)(0)(0) - 0.5) < 0.000000001)
    assert(abs(scheffeIntegrals(1)(0)(1) - 1.0) < 0.000000001)

    deltas = new Array(k)
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

    mdeIndex = 0 
    for (i <- 1 until k) {
      if (deltas(i) < deltas(mdeIndex)) {
        mdeIndex = i
      }
    }

    assert(abs(deltas(0) - 0.0) < 0.000000001)
    assert(abs(deltas(1) - 0.5) < 0.000000001)
    assert(abs(deltas(mdeIndex) - 0.0) < 0.000000001)
  }

  it should "produce the correct result in another simple example" in {
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
    var crpMaxDepth = 0
    for (i <- 0 until crpLeaves.length) {
      for (j <- 0 until k) {
        crpValues(i)(k-1-j) = crp.densities.vals(i).apply(s"$j")._1
      }
      crpLeaves(i) = (crp.densities.truncation.leaves(i), crpValues(i))
      crpMaxDepth = max(crpMaxDepth, crp.densities.truncation.leaves(i).depth)
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
    
    val mergedValidationData = validationData.mapPartitions(iter => scheffeSetsValidationCount(crpLeaves, iter, crpMaxDepth, scheffeCountAccumulators)).cache

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

    val maxCrpDepth = crp.densities.leaves.map(_.depth).max
    val crpLeafSet = crp.densities.leaves.toSet
    val crpLeafMap = crp.densities.copy(vals = Stream.continually(0).take(crp.densities.leaves.length).toVector)

    val truncatedValData = validationData.map(t => (t._1.truncate(maxCrpDepth), t._2)).reduceByKey{(v1,v2) => v1 + v2}
    val valHist = Histogram(
      tree,
      truncatedValData.map(_._2).reduce(_+_),
      fromNodeLabelMap(
        { leafMap: LeafMap[_] =>
            truncatedValData.map(t => { (findSubtree(t._1, leafMap.truncation.leaves), t._2) }).reduceByKey((v1, v2) => v1 + v2)
        }.apply(crpLeafMap)
          .collect.toMap
      )
    )

    val validationDeviations = getDelta(crp, valHist)

    val bestDelta = validationDeviations.head._2
    assert(abs(bestDelta - 0.05) > 0.0001)
    println("getMDEPrime contains bug in calculation of Delta: | correctDelta - caluclateDelta | = " + abs(bestDelta - 0.05))
  }
}
