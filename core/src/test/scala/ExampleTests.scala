import scala.language.postfixOps


import org.apache.spark._
import org.apache.log4j.{ Logger, Level }
import org.apache.spark.mllib.linalg.{ Vector => MLVector, _ }
import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.sql
import org.apache.spark.rdd.RDD

import org.scalatest.{ path => testPath, _ }
import org.scalactic.TolerantNumerics
import org.scalactic.TripleEquals._

import scala.math.abs
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashMap._

import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.RectangleFunctions._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.GslRngFunctions._
import co.wiklund.disthist.LeafMapFunctions._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.TruncationFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.SubtreePartitionerFunctions._

import org.apache.spark.mllib.random.RandomRDDs.uniformVectorRDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
    
case class Song(year : Int, timbres : Vector[Double]) extends Serializable

class ExampleTests extends FlatSpec with Matchers with BeforeAndAfterAll {

  override protected def beforeAll: Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  override protected def afterAll: Unit = {
    val spark = getSpark
    spark.stop
  }
  
  implicit val ordering : Ordering[NodeLabel] = leftRightOrd

  var density : DensityHistogram = null

  "Slice and marginal" should "behave as expected" in {
    val spark = getSpark
    import spark.implicits._
  
    val dimensions = 2
    val sizeExp = 5
    val numPartitions = 8
    spark.conf.set("spark.default.parallelism", numPartitions.toString)
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-5 
  
    val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
    val rawTestRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)
  
    /* Get boxhull of training data and test data */
    var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
    var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectTest)
  
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
    val kInMDE = 10
  
    /**
     * Label each datapoint by the label of the box at depth (finestResDepth) which it resides in, and count number of datapoints
     * found in occupied boxes.
     */
    var countedTrain = quickToLabeled(tree, finestResDepth, rawTrainRDD)
    var countedTest = quickToLabeled(tree, finestResDepth, rawTestRDD)
        
    val sampleSizeHint = 20
    val partitioner = new SubtreePartitioner(numPartitions, countedTrain, sampleSizeHint)
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 200 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit)
  
    val mde = getMDE(
      Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap)), 
      countedTest, 
      kInMDE, 
      false 
    )

    /**
     * Conditional Density - We need a densityHistogram or Histogram. Furthermore, we need a vector of doubles representing the values
     * of the axes we which to slice. The slice is done orthogonal to the axes we choose, at the given vector.
     */
    val slicePoints : Vector[Double] = Vector(1.0)

    /* Note: We give indices here, 0 == First axis, 1 == Second axis and so on... */
    val sliceAxes : Vector[Axis] = Vector(0) 

    density = toDensityHistogram(mde)
    var conditional = slice(density, sliceAxes, slicePoints)
    var prob = conditional.densityMap.vals.map{ case (dens, vol) => dens * vol}.sum
    println("Slice integral value: " + prob)

    /* We need to normalize the retrieved conditional */
    conditional = conditional.normalize
    prob = conditional.densityMap.vals.map{ case (dens, vol) => dens * vol}.sum
    println("Normalized integral value: " + prob)

    /**
     * Marginal Density - We need a densityHistogram or Histogram. We also need a vector axesToKeep which defines the axes to
     * keep, i.e. the dimensions which the density will vary over after being marginalized.
     */
    val axesToKeep = Vector(0)
    val marginal = marginalize(density, axesToKeep)
    var margProb = marginal.densityMap.vals.map{ case (dens, vol) => dens * vol}.sum
    println("Marginal integral value: " + margProb)
  }

  "Sampling" should "behave as expected" in {
    val spark = getSpark
    import spark.implicits._

    /************ Sampling from GslRngHandle ************/

    /* Allocate Gnu Scientific Library RNG using given seed */
    val rngHandle = new GslRngHandle(1235)

    println("gsl_rng address: " + rngHandle.gslRngAddress)

    val probabilities : Array[Double] = Array(0.1, 0.2, 0.3, 0.4)
    var outcomes : Array[Double] = Array(0,0,0,0)
    val sampleSize = 10000
    val sample1 = gsl_ran_discrete_fill_buffer(rngHandle.gslRngAddress, probabilities, sampleSize)
    sample1.foreach(i => outcomes(i) += 1)
    println("Outcome Discrete distribution with probabilities [0.1 0.2 0.3 0.4]: ")
    println(outcomes(0) / sampleSize)
    println(outcomes(1) / sampleSize)
    println(outcomes(2) / sampleSize)
    println(outcomes(3) / sampleSize)

    val flatSample = gsl_ran_flat_fill_buffer(rngHandle.gslRngAddress, 3.0, 9.0, sampleSize)
    println("Uniform[3.0, 9.0] empirical mean: " + flatSample.sum / sampleSize)

    /************ Sampling from DensityHistogram ************/

    val sample2 = density.sample(rngHandle, sampleSize)
    sample2.foreach(println(_))

    /************ Free Resources ************/
    rngHandle.free
    println("RNG Freed!")
  }
}

class SongExamples extends FlatSpec with Matchers with BeforeAndAfterAll {

  override protected def beforeAll: Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  override protected def afterAll: Unit = {
    val spark = getSpark
    spark.stop
  }
  
  implicit val ordering : Ordering[NodeLabel] = leftRightOrd

  "Milion Song Dataset" should "work" in {
    val spark = getSpark
    import spark.implicits._
 
    def toDouble(token : String, default : Double) : Double = {
      try {
        token.toDouble 
      } catch {
        case e : Exception => default
      }
    }
    
    def toInt(token : String, default : Int) : Int = {
      try {
        token.toInt 
      } catch {
        case e : Exception => default
      }
    }

    /* Load in Data and create RDD of Songs(year, Vector(timbre1,...,timbre12)) */
    val trainDataRDD = spark.sparkContext.textFile("../datasets/TrainingYearPredictionMSD.txt")
    val trainSongRDD = trainDataRDD.map(line => { 
      val tokens = line.split(",")
      Vectors.dense(toInt(tokens(0), 0),
         toDouble(tokens(1), 0.0) 
        ,toDouble(tokens(2), 0.0)
        ,toDouble(tokens(3), 0.0)
        ,toDouble(tokens(4), 0.0)
        ,toDouble(tokens(5), 0.0)
        ,toDouble(tokens(6), 0.0)
        ,toDouble(tokens(7), 0.0)
        ,toDouble(tokens(8), 0.0)
        ,toDouble(tokens(9), 0.0)
        ,toDouble(tokens(10), 0.0)
        ,toDouble(tokens(11), 0.0)
        ,toDouble(tokens(12), 0.0)
      )
    })

    val testDataRDD = spark.sparkContext.textFile("../datasets/TestYearPredictionMSD.txt")
    val testRDD = testDataRDD.map(line => { 
      val tokens = line.split(",")
      Vectors.dense(toDouble(tokens(0), 0),
         toDouble(tokens(1), 0.0) 
        ,toDouble(tokens(2), 0.0)
        ,toDouble(tokens(3), 0.0)
        ,toDouble(tokens(4), 0.0)
        ,toDouble(tokens(5), 0.0)
        ,toDouble(tokens(6), 0.0)
        ,toDouble(tokens(7), 0.0)
        ,toDouble(tokens(8), 0.0)
        ,toDouble(tokens(9), 0.0)
        ,toDouble(tokens(10), 0.0)
        ,toDouble(tokens(11), 0.0)
        ,toDouble(tokens(12), 0.0)
      )
    })

    /* Split into training set, validation set and test set */
    val trainingProportion = 0.66
    val rdds = trainSongRDD.randomSplit(Array(trainingProportion, 1.0 - trainingProportion))

    /* Data used to construct the estimator's partitioning of the data space */
   val trainingRDD = rdds(0)

    /* Used to determine what partitioning created from the trainingRDD performs best (constructing the minimum distance estimate) */
    val validationRDD = rdds(1) 

    val dimensions = 13

    /**
     * (Technical) : TODO 
     */
    val numPartitions = 8
    spark.conf.set("spark.default.parallelism", numPartitions.toString)

    /**
     * (Technical) : TODO 
     */
    val finestResSideLength = 1e-2 
  
    /* Get boxhull of training data and test data */
    val rectTrain = RectangleFunctions.boundingBox(trainingRDD)
    val rectValidation = RectangleFunctions.boundingBox(validationRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)
  
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
    val kInMDE = 10
  
    /**
     * Label each datapoint by the label of the box at depth (finestResDepth) which it resides in, and count number of datapoints
     * found in occupied boxes.
     */
    val countedTrain = quickToLabeled(tree, finestResDepth, trainingRDD)
    val countedValidation = quickToLabeled(tree, finestResDepth, validationRDD)
        
    /**
     * (Technical) : sample size hint gives the subtree partitioner a hint of how large of a sample it should sample from countedTrain
     *               from which it then an estimate of how the data is distributed among the branches. 
     */
    val sampleSizeHint = 20
    val partitioner = new SubtreePartitioner(numPartitions, countedTrain, sampleSizeHint)
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 100 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit)
  
    val density = toDensityHistogram(getMDE(
      Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap)), 
      countedValidation, 
      kInMDE, 
      false 
    ))

    /* Allocate Gnu Scientific Library RNG using given seed */
    val rngHandle = new GslRngHandle(1235)

    val testData = testRDD.collect  
    
    var wantedConfidence = 0.95
    var withinCoverageRegion : Int = 0
    //for (i <- 0 until 1000) {
    for (i <- 0 until testData.length) {

      /* TODO: Why is slice written for Vector[double] while mergestuff happens in MLVector ... very bad this */
      /* Drop year, we wish to condition on the timbres */
      val slicePoint : Vector[Double] = testData(i).toArray.toVector.drop(1) 
  
      /* Note: We give indices here, 0 == First axis, 1 == Second axis and so on... */
      val sliceAxes : Vector[Axis] = Vector(1,2,3,4,5,6,7,8,9,10,11,12) 

      val tree : WidestSplitTree = WidestSplitTree(density.tree.rootCell) 
      val splitOrder = tree.splitOrderToDepth(finestResDepth)
      var sliceLeavesBuf : Array[NodeLabel] = new Array(density.densityMap.truncation.leaves.length)
      var sliceValuesBuf : Array[(Probability,Volume)] = new Array(density.densityMap.truncation.leaves.length)
      var conditional = quickSlice(density, sliceAxes, slicePoint, splitOrder)

      /* Conditional Density */
      //val conditionalOld = slice(density, sliceAxes, slicePoint).normalize
      //val leaves1 = conditional.densityMap.truncation.leaves
      //val leaves2 = conditionalOld.densityMap.truncation.leaves
      //val vals1 = conditional.densityMap.vals
      //val vals2 = conditionalOld.densityMap.vals

      //println("LENGTHS: " + leaves1.length + ", " + leaves2.length)
      //var prob1 : Double = 0
      //var prob2 : Double = 0
      //for (j <- 0 until leaves1.length) {
      //  println("leaves: " + leaves1(j) + ", " + leaves2(j))
      //  println("values: " + vals1(j) + ", " + vals2(j))
      //  prob1 += vals1(j)._1 * vals1(j)._2
      //  prob2 += vals2(j)._1 * vals2(j)._2
      //  assert(leaves1(j) == leaves2(j))
      //}
      //assert(leaves1.length == leaves2.length)

      //for (j <- 1 to 100) {
      //  println("Slice: " + j)
      //  val cawd = slice(density, sliceAxes, slicePoints)
      //}

      //for (j <- 0 until testData.length) {
      //  println("Quick slice " + j)
      //  val cawd = quickSlice(density, sliceAxes, testData(j).toArray.toVector.drop(1), splitOrder, sliceLeavesBuf, sliceValuesBuf)
      //}
      //
      

      if (conditional != null) {
        conditional = conditional.normalize
        /**
         * Get a Map of leaf -> last element in Coverage Region.  If we wish to determine whether a point V found in leaf L
         * exist in the coverage region of confidence 0 < C <= 1, we simply check if Map(L) <= C. If the statement is true,
         * V was found to be within the coverage region of confidence C.
         */
        val coverageRegions = conditional.tailProbabilities

        /* Retrieve confidence region */
        val coverageRegionConfidence  = coverageRegions.query(Vectors.dense(Array((testData(i))(0))))

        if (coverageRegionConfidence <= wantedConfidence) { 
          withinCoverageRegion += 1
        }
        println("Confidence, withinConfidenceRegion:  " + wantedConfidence + ", " + withinCoverageRegion.toDouble / (i+1))

        /* Generate a sample from the conditional distribution */
        if (conditional.densityMap.truncation.leaves.length > 0) {
          val prediction = ((conditional.sample(rngHandle, 1))(0))(0)
          //println("TRUE, PREDICTED: " + (testData(i))(0) + ", " + prediction)
        }
      }
    }
  
    /* Free Resources */
    rngHandle.free
  }
}

class NormalExamples extends FlatSpec with Matchers with BeforeAndAfterAll {

  override protected def beforeAll: Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  override protected def afterAll: Unit = {
    val spark = getSpark
    spark.stop
  }
  
  implicit val ordering : Ordering[NodeLabel] = leftRightOrd

  "Normal Regression Analysis" should "work" in {
    val spark = getSpark
    import spark.implicits._

    /**
     * TODO: What is this estimator, link papers, explain basic ideas, data structures.
     * - Quick description of regular paving, (details: [Mapped Regular Pavings]
     * - [Insert Latex Picture]
     * - Histogram estimator main idea. 
     */
 
    /**
     * TODO: Why use estimator (What is its strenghts)
     * - Works in L_1, easily understandable evaluation of performance
     * - Works for any continuous X in L_1, with universal performance guarantees.
     * - Works for any dimension
     * - Made to be scalable, constructed for distributed data sets.
     * - The Strict arithmetic whis emerges from Regular Paving splitting rules allow us to easily construct
     *   useful tools: We can condition on arbitrary points in arbitrary dimensions. From that we can easily retrieve
     *   confidence intervals.
     */

    /**
     * TODO: Weaknesses 
     * - Hard to use. (Several parameters which we can adjust to our need, requires attention / responsible use by user.
     * - [Technical] Somewhat error-prone (We can mitigate this!)
     * - constructed in 4 stages, 3 of which the user has to get their hands dirty in.
     */

    /**
     * Setup of problem data. We consider here data from a 5-dimensional standard Gaussian. We will train the estimator
     * using a training size of 10^7, a validation size of (1/2) * 10^7, and a test size of 1000 to evaluate the performance
     * of our estimator. The number of partitions is an important parameter which should not necessarily be set to the number of
     * available cores over all machines, we shall get into this later.
     */
    val dimensions = 5
    val sizeExp = 7
    val numPartitions = 64
    spark.conf.set("spark.default.parallelism", numPartitions.toString)
    val trainSize = math.pow(10, sizeExp).toLong

  
    /**
     * We now generate all the data.
     */
    val trainingRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1230568)
    val validationRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 5465694)
    val testRDD =  normalVectorRDD(spark.sparkContext, 1000, dimensions, numPartitions, 6949239)

  
    /**
     * Stage 1: TODO
     */

    /* Get boxhull of training data and test data */
    var rectTrain = RectangleFunctions.boundingBox(trainingRDD)
    var rectValidation = RectangleFunctions.boundingBox(validationRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)
  
    /**
     * finestResSideLength: TODO
     */
    val finestResSideLength = 1e-5 
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
  
    /**
     * TODO: Stage 2: Labeling of data at finestResDepth
     */
    var countedTrain = quickToLabeled(tree, finestResDepth, trainingRDD)
    var countedValidation = quickToLabeled(tree, finestResDepth, validationRDD)
        
    /**
     * TODO: Stage 3: Sort data according to subtrees they are found in, merge them up to count limit
     */
    val sampleSizeHint = 1000
    val partitioner = new SubtreePartitioner(numPartitions, countedTrain, sampleSizeHint)
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 400 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit)

    /**
     * TODO: Stage 4: Finding a ~MDE between our count limit and the rootBox density
     */
    val kInMDE = 10
    val density = toDensityHistogram(getMDE(
      Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap)), 
      countedValidation, 
      kInMDE, 
      true 
    ))

    /**
     * TODO: we finally have our non-normalized density.
     *  (1) Normalize
     *  (2) Go through the density's data structures
     *  (3) Visualize
     *  (4) Write up save/get functions 
     */

/********************** Notebook 2 *********************/

    /* Allocate Gnu Scientific Library RNG using given seed */
    val rngHandle = new GslRngHandle(1235)

    val testData = testRDD.collect  
    
    var wantedConfidence = 0.95
    var withinCoverageRegion : Int = 0
    for (i <- 0 until testData.length) {

      /* TODO: Why is slice written for Vector[double] while mergestuff happens in MLVector ... very bad this */
      /* Drop year, we wish to condition on the timbres */
      val slicePoint : Vector[Double] = testData(i).toArray.toVector.drop(1) 
  
      /* Note: We give indices here, 0 == First axis, 1 == Second axis and so on... */
      val sliceAxes : Vector[Axis] = Vector(1,2,3,4) 

      val tree : WidestSplitTree = WidestSplitTree(density.tree.rootCell) 
      val splitOrder = tree.splitOrderToDepth(finestResDepth)
      var sliceLeavesBuf : Array[NodeLabel] = new Array(density.densityMap.truncation.leaves.length)
      var sliceValuesBuf : Array[(Probability,Volume)] = new Array(density.densityMap.truncation.leaves.length)
      var conditional = quickSlice(density, sliceAxes, slicePoint, splitOrder)

      if (conditional != null) {
        conditional = conditional.normalize
        /**
         * Get a Map of leaf -> last element in Coverage Region.  If we wish to determine whether a point V found in leaf L
         * exist in the coverage region of confidence 0 < C <= 1, we simply check if Map(L) <= C. If the statement is true,
         * V was found to be within the coverage region of confidence C.
         */
        val coverageRegions = conditional.tailProbabilities

        /* Retrieve confidence region */
        val coverageRegionConfidence  = coverageRegions.query(Vectors.dense(Array((testData(i))(0))))

        if (coverageRegionConfidence <= wantedConfidence) { 
          withinCoverageRegion += 1
        }
        println("Confidence, withinConfidenceRegion:  " + wantedConfidence + ", " + withinCoverageRegion.toDouble / (i+1))

        /* Generate a sample from the conditional distribution */
        if (conditional.densityMap.truncation.leaves.length > 0) {
          val prediction = ((conditional.sample(rngHandle, 1))(0))(0)
          //println("TRUE, PREDICTED: " + (testData(i))(0) + ", " + prediction)
        }
      }
    }
  
    /* Free Resources */
    rngHandle.free
  }
}
