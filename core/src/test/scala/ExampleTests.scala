import scala.language.postfixOps

import co.wiklund.disthist._
import SpatialTreeFunctions._
import RectangleFunctions._
import MergeEstimatorFunctions._
import HistogramFunctions._
import LeafMapFunctions._
import MDEFunctions._
import GslRngFunctions._
import Types._

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
    for (i <- 0 until 1000) {
    //for (i <- 0 until testData.length) {

      /* TODO: Why is slice written for Vector[double] while mergestuff happens in MLVector ... very bad this */
      /* Drop year, we wish to condition on the timbres */
      val slicePoints : Vector[Double] = testData(i).toArray.toVector.drop(1) 
  
      /* Note: We give indices here, 0 == First axis, 1 == Second axis and so on... */
      val sliceAxes : Vector[Axis] = Vector(1,2,3,4,5,6,7,8,9,10,11,12) 

      /* TODO: What happens if slicePoint is not within rootBox? Possible bug... */ 
      /* Conditional Density */
      val conditional = slice(density, sliceAxes, slicePoints).normalize

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
      println("Confidence, withinConfidenceRegion: " + wantedConfidence + ", " + withinCoverageRegion.toDouble / (i+1))

      /* Generate a sample from the conditional distribution */
      val prediction = ((conditional.sample(rngHandle, 1))(0))(0)
      println("TRUE, PREDICTED: " + (testData(i))(0) + ", " + prediction)
    }
  
    /* Free Resources */
    rngHandle.free
  }
}
