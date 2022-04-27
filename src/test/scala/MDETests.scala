import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD

import org.apache.log4j.{ Logger, Level }

import scala.reflect.io.Directory
import java.io.File

import co.wiklund.disthist._
import MDEFunctions._
import MergeEstimatorFunctions._
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
  private val tree = widestSideTreeRootedAt(rootBox)

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

  "getMDE" should "give a correct histogram" in {
    val spark = getSpark
    import spark.implicits._
    val mergedHist = collectHistogram(tree, spark.read.parquet(checkpointDir + "/merged").as[(NodeLabel, Count)])
    val mdeHist = getMDE(mergedHist, valDS, 10)

    mdeHist.counts.vals.sum shouldEqual dfnum
  }
}
