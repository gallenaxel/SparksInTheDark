import scala.language.postfixOps

import co.wiklund.disthist._
import SpatialTreeFunctions._
import RectangleFunctions._
import MergeEstimatorFunctions._
import HistogramFunctions._
import LeafMapFunctions._
import MDEFunctions._
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
//import co.wiklund.disthist.SubtreePartitionerFunctions._
import co.wiklund.disthist.SubtreePartitionerFunctions._

import org.apache.spark.mllib.random.RandomRDDs.uniformVectorRDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
    
class SubtreePartitioningTests extends FlatSpec with Matchers with BeforeAndAfterAll {

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
  
  val lrOrd : Ordering[NodeLabel] = leftRightOrd

  "leftRightOrdering" should "Generate a intuitive left to right ordering of NodeLabel leaves" in {
    val LRCopy = NodeLabel(5)
    val LR = NodeLabel(5)
    val LL = NodeLabel(4)
    val R = NodeLabel(3)
    assert(lrOrd.compare(LR, LRCopy)  ==  0)
    assert(lrOrd.compare(LRCopy, LR)  ==  0)
    assert(lrOrd.compare(LL, LR)  == -1)
    assert(lrOrd.compare(LR, LL)  ==  1)
    assert(lrOrd.compare(LL, R)  ==  -1)
    assert(lrOrd.compare(LR, R)  ==  -1)
    assert(lrOrd.compare(R, LL)  ==   1)
    assert(lrOrd.compare(R, LR)  ==   1)
  }

  "RangeParitioning" should "locally sort each partition according to leftRightOrdering" in {
     val spark = getSpark
     import spark.implicits._

     val labels = Vector(16,17,18,19,21,22,24,25,26,27,15).map(NodeLabel(_)).map((_,1))
     val numParititions = 4
     val labelsRDD = spark.sparkContext.parallelize(labels, numParititions)
     val labelsDS = spark.createDataset(labelsRDD)
     implicit val ordering : Ordering[NodeLabel] = leftRightOrd
     val rangeDS = labelsDS.repartitionByRange(4, $"_1")

    val localRanges = rangeDS.mapPartitions(iter => {
      val vec = iter.toVector.map(t => t._1)
      Vector(vec).toIterator
    }).collect    

    val isLocallySorted = localRanges.map(vec => {
      var isSorted : Boolean = true
      for (i <- 0 until vec.length) {
        for (j <- (i+1) until vec.length) {
          if (lrOrd.compare(vec(i), vec(j)) == -1) {
            isSorted = isSorted && true
          } else {
            isSorted = isSorted && false
          }
        }
      }
      isSorted
    })

    isLocallySorted.foreach(assert(_))
  }
  
  it should "be globally sorted, i.e. each partition contains a contiguous range in the whole sorted collection" in {
    val spark = getSpark
    import spark.implicits._

    /* True Order: 16,17,18,19,21,22,24,25,26,27,15 */
    val labels = Vector(19,25,17,21,15,24,18,26,16,27,22).map(NodeLabel(_)).map((_,1))
    val numParititions = 4
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
    val labelsRDD : RDD[(NodeLabel, Int)] = spark.sparkContext.parallelize(labels, numParititions)

    val sortedRDD = labelsRDD.sortByKey(ascending = true, 4)
    //println("---- RDD sorting ----")
    //sortedRDD.glom.collect.zipWithIndex.foreach(t => {
    //  println("Partition " + t._2 + ":")
    //  t._1.foreach(println(_))
    //})

    val groupRDD = labelsRDD.sortByKey(ascending = true, 4).groupByKey()
    //println("\n---- RDD sorting + GroupByKey ----")
    //groupRDD.mapPartitions(iter => {Vector(iter.toVector).toIterator} ).collect.foreach(println(_))

    val globalRanges = sortedRDD.map(t => t._1).collect
    var isGlobalSorted : Boolean = true
    //println("\n---- RDD sorting + collect ----")
    //globalRanges.foreach(println(_))
    for (i <- 0 until globalRanges.length) {
      for (j <- (i+1) until globalRanges.length) {
        if (lrOrd.compare(globalRanges(i), globalRanges(j)) == -1) {
          isGlobalSorted = isGlobalSorted && true
        } else {
          isGlobalSorted = isGlobalSorted && false
        }
      }
    }
    assert(isGlobalSorted)
  }

  val leaves1 = Vector((16,1.0f),(17,1.0f),(18,1.0f),(19,1.0f),(23,1.0f),(24,1.0f),(25,1.0f),(26,1.0f),(27,1.0f),(28,1.0f),(29,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val maxWeight1 : Float = 4.0f
    val subtrees1 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves1, maxWeight1)

    val leaves2 = Vector((16,1.0f),(17,1.0f),(18,1.0f),(19,1.0f),(23,1.0f),(24,1.0f),(25,1.0f),(26,1.0f),(27,1.0f),(28,1.0f),(29,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val maxWeight2 : Float = 11.0f
    val subtrees2 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves2, maxWeight2)

    val leaves3 = Vector((42,1.0f),(86,1.0f),(87,1.0f),(88,1.0f),(89,1.0f),(90,1.0f),(91,1.0f),(24,1.0f),(25,1.0f),(13,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val maxWeight3 : Float = 2.0f
    val subtrees3 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves3, maxWeight3)

    val leaves4 = Vector((33,1.0f),(76,1.0f),(77,1.0f),(82,1.0f),(83,1.0f),(92,1.0f),(6,1.0f),(56,1.0f),(57,1.0f),(58,1.0f),(59,1.0f),(30,1.0f),(62,1.0f),(63,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val maxWeight4 : Float = 3.0f
    val subtrees4 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves4, maxWeight4)

    val leaves5 = Vector((33,1.0f),(76,1.0f),(77,1.0f),(82,1.0f),(83,1.0f),(92,1.0f),(6,1.0f),(56,1.0f),(57,1.0f),(58,1.0f),(59,1.0f),(30,1.0f),(62,1.0f),(63,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val maxWeight5 : Float = 20.0f
    val subtrees5 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves5, maxWeight5)

  "maximalSubtreeGeneration" should "generate correct subtrees" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    assert(subtrees2.length == 1)
    assert(subtrees2(0)._1.lab == 1 && subtrees2(0)._2 == 11)
    
    assert(subtrees1.length == 4)
    assert(subtrees1(0)._1.lab == 4 && subtrees1(0)._2 == 4)
    assert(subtrees1(1)._1.lab == 5 && subtrees1(1)._2 == 1)
    assert(subtrees1(2)._1.lab == 6 && subtrees1(2)._2 == 4)
    assert(subtrees1(3)._1.lab == 7 && subtrees1(3)._2 == 2)

    assert(subtrees3.length == 6)
    assert(subtrees3(0)._1.lab == 42 && subtrees3(0)._2 == 1)
    assert(subtrees3(1)._1.lab == 43 && subtrees3(1)._2 == 2)
    assert(subtrees3(2)._1.lab == 44 && subtrees3(2)._2 == 2)
    assert(subtrees3(3)._1.lab == 45 && subtrees3(3)._2 == 2)
    assert(subtrees3(4)._1.lab == 12 && subtrees3(4)._2 == 2)
    assert(subtrees3(5)._1.lab == 13 && subtrees3(5)._2 == 1)

    assert(subtrees4.length == 6)
    assert(subtrees4(0)._1.lab == 4  && subtrees4(0)._2 == 3)
    assert(subtrees4(1)._1.lab == 5  && subtrees4(1)._2 == 3)
    assert(subtrees4(2)._1.lab == 6  && subtrees4(2)._2 == 1)
    assert(subtrees4(3)._1.lab == 28 && subtrees4(3)._2 == 2)
    assert(subtrees4(4)._1.lab == 29 && subtrees4(4)._2 == 2)
    assert(subtrees4(5)._1.lab == 15 && subtrees4(5)._2 == 3)

    assert(subtrees5.length == 1)
    assert(subtrees5(0)._1.lab == 1 && subtrees5(0)._2 == 14)
  }

  def spmToWeightVec(spm : HashMap[NodeLabel, Int], subtrees : Array[(NodeLabel, Float)]) : Vector[(Int, Float)] = {
    val vec : Vector[(Int, Float)] = 
    spm
    .toVector
    .groupBy(t => t._2)
    .toVector
    .map(kt => {
      val weight = kt._2.map(t => t._1)
        .map(leaf => {
          var f : Float = 0.0f
          var bool : Boolean = true
          var i : Int = 0
          while (bool) {
            if (subtrees(i)._1.lab == leaf.lab) {
              f = subtrees(i)._2 
              bool = false 
            }
            i += 1
          }
          f
        }).reduce(_+_) 
      (kt._1, weight)
    })
    vec
   }

  "distributeSubtreesToPartitions" should "setup correct subtree -> partition map" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val spm1 = distributeSubtreesToPartitions(2, subtrees1)
    val countPart1 = spmToWeightVec(spm1, subtrees1).map(t => t._2).sorted
    assert(countPart1 == Vector(5L,6L))

    val spm2 = distributeSubtreesToPartitions(4, subtrees2)
    val countPart2 = spmToWeightVec(spm2, subtrees2).map(t => t._2).sorted
    assert(countPart2 == Vector(11L))

    val spm3 = distributeSubtreesToPartitions(1, subtrees3)
    val countPart3 = spmToWeightVec(spm3, subtrees3).map(t => t._2).sorted
    assert(countPart3 == Vector(10L))

    val spm4 = distributeSubtreesToPartitions(2, subtrees3)
    val countPart4 = spmToWeightVec(spm4, subtrees3).map(t => t._2).sorted
    assert(countPart4 == Vector(5L,5L))

    val spm5 = distributeSubtreesToPartitions(3, subtrees3)
    val countPart5 = spmToWeightVec(spm5, subtrees3).map(t => t._2).sorted
    assert(countPart5 == Vector(3L,3L,4L))

    val spm6 = distributeSubtreesToPartitions(5, subtrees3)
    val countPart6 = spmToWeightVec(spm6, subtrees3).map(t => t._2).sorted
    assert(countPart6 == Vector(2L,2L,2L,2L,2L))

    val spm7 = distributeSubtreesToPartitions(2, subtrees4)
    val countPart7 = spmToWeightVec(spm7, subtrees4).map(t => t._2).sorted
    assert(countPart7 == Vector(7L,7L))

    val spm8 = distributeSubtreesToPartitions(3, subtrees4)
    val countPart8 = spmToWeightVec(spm8, subtrees4).map(t => t._2).sorted
    assert(countPart8 == Vector(4L,5L,5L))

    val spm9 = distributeSubtreesToPartitions(9, subtrees4)
    val countPart9 = spmToWeightVec(spm9, subtrees4).map(t => t._2).sorted
    assert(countPart9 == Vector(1L,2L,2L,3L,3L,3L))
  }

  "SubtreePartitioner" should "find the correct subtree for each leaf" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val truncWithoutSubtree = Truncation(Vector(4,8,20,40,23,46,7,15).map(NodeLabel(_)))
    val subtrees = subtrees3
    assert(findSubtree(NodeLabel(42), subtrees) ==  NodeLabel(42))
    assert(findSubtree(NodeLabel(84), subtrees) ==  NodeLabel(42))
    assert(findSubtree(NodeLabel(86), subtrees) ==  NodeLabel(43))
    assert(findSubtree(NodeLabel(87), subtrees) ==  NodeLabel(43))
    assert(findSubtree(NodeLabel(172), subtrees) == NodeLabel(43))
    assert(findSubtree(NodeLabel(88), subtrees) ==  NodeLabel(44))
    assert(findSubtree(NodeLabel(89), subtrees) ==  NodeLabel(44))
    assert(findSubtree(NodeLabel(90), subtrees) ==  NodeLabel(45))
    assert(findSubtree(NodeLabel(91), subtrees) ==  NodeLabel(45))
    assert(findSubtree(NodeLabel(24), subtrees) ==  NodeLabel(12))
    assert(findSubtree(NodeLabel(25), subtrees) ==  NodeLabel(12))
    assert(findSubtree(NodeLabel(48), subtrees) ==  NodeLabel(12))
    assert(findSubtree(NodeLabel(13), subtrees) ==  NodeLabel(13))
    assert(findSubtree(NodeLabel(26), subtrees) ==  NodeLabel(13))
    assert(findSubtree(truncWithoutSubtree.leaves(0), subtrees) == NodeLabel(42))
    assert(findSubtree(truncWithoutSubtree.leaves(1), subtrees) == NodeLabel(42))
    assert(findSubtree(truncWithoutSubtree.leaves(2), subtrees) == NodeLabel(42))
    assert(findSubtree(truncWithoutSubtree.leaves(3), subtrees) == NodeLabel(42))
    assert(findSubtree(truncWithoutSubtree.leaves(4), subtrees) == NodeLabel(45))
    assert(findSubtree(truncWithoutSubtree.leaves(5), subtrees) == NodeLabel(45))
    assert(findSubtree(truncWithoutSubtree.leaves(6), subtrees) == NodeLabel(13))
    assert(findSubtree(truncWithoutSubtree.leaves(7), subtrees) == NodeLabel(13))
  } 

  it should "send each leaf to the correct partition" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val truncWithoutSubtree = Truncation(Vector(4,20,46,94,7).map(NodeLabel(_)))
    val sampleLeaves = Vector(42,86,87,88,89,90,91,24,25,13).map(l => (NodeLabel(l), 1L))
    val allLeaves = Vector(4,20,46,94,7,42,86,87,88,89,90,91,24,25,13).map(l => (NodeLabel(l), 1L))
    val sampleRDD = spark.sparkContext.parallelize(sampleLeaves, 1)
    val subtreePartitioner = new SubtreePartitioner(4, sampleRDD, 10)
    val fullRDD = spark.sparkContext.parallelize(allLeaves, 4)
    val subtreeRDD = fullRDD.repartitionAndSortWithinPartitions(subtreePartitioner)

    assert(subtreePartitioner.getPartition(NodeLabel(42)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(42)))
    assert(subtreePartitioner.getPartition(NodeLabel(84)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(42)))
    assert(subtreePartitioner.getPartition(NodeLabel(86)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(43)))
    assert(subtreePartitioner.getPartition(NodeLabel(87)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(43)))
    assert(subtreePartitioner.getPartition(NodeLabel(172)) == subtreePartitioner.subtreePartitionMap(NodeLabel(43)))
    assert(subtreePartitioner.getPartition(NodeLabel(88)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(44)))
    assert(subtreePartitioner.getPartition(NodeLabel(89)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(44)))
    assert(subtreePartitioner.getPartition(NodeLabel(90)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(45)))
    assert(subtreePartitioner.getPartition(NodeLabel(91)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(45)))
    assert(subtreePartitioner.getPartition(NodeLabel(24)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(12)))
    assert(subtreePartitioner.getPartition(NodeLabel(25)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(12)))
    assert(subtreePartitioner.getPartition(NodeLabel(48)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(12)))
    assert(subtreePartitioner.getPartition(NodeLabel(13)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(13)))
    assert(subtreePartitioner.getPartition(NodeLabel(26)) ==  subtreePartitioner.subtreePartitionMap(NodeLabel(13)))
    assert(subtreePartitioner.getPartition(truncWithoutSubtree.leaves(0)) == subtreePartitioner.subtreePartitionMap(NodeLabel(42))) 
    assert(subtreePartitioner.getPartition(truncWithoutSubtree.leaves(1)) == subtreePartitioner.subtreePartitionMap(NodeLabel(42)))
    assert(subtreePartitioner.getPartition(truncWithoutSubtree.leaves(2)) == subtreePartitioner.subtreePartitionMap(NodeLabel(45)))
    assert(subtreePartitioner.getPartition(truncWithoutSubtree.leaves(3)) == subtreePartitioner.subtreePartitionMap(NodeLabel(45)))
    assert(subtreePartitioner.getPartition(truncWithoutSubtree.leaves(4)) == subtreePartitioner.subtreePartitionMap(NodeLabel(13)))

    subtreeRDD.glom().collect().foreach(arr => {
      val vec = arr.toVector
      var i = 0
      while (i < vec.length) {
        var j = i + 1
        while (j < vec.length) {
          assert(leftRightOrd.compare(vec(i)._1, vec(j)._1) == -1)
          j += 1
        }
        i += 1
      }
    })

    val trunc5 = Truncation(Vector(33,76,77,82,83,92,56,57,58,59,30,62,63).map(NodeLabel(_)))
    val maxCount5 : Count = 3
    val truncWithoutSubtree5 = Truncation(Vector(12,13).map(NodeLabel(_)))
    val sampleLeaves5 = Vector(33,76,77,82,83,92,56,57,58,59,30,62,63).map(l => (NodeLabel(l), 1L))
    val allLeaves5 = Vector(12,13,33,76,77,82,83,92,56,57,58,59,30,62,63).map(l => (NodeLabel(l), 1L))
    val sampleRDD5 = spark.sparkContext.parallelize(sampleLeaves5, 1)
    val subtreePartitioner5 = new SubtreePartitioner(4, sampleRDD5, 13)
    val fullRDD5 = spark.sparkContext.parallelize(allLeaves5, 4)
    val subtreeRDD5 = fullRDD5.repartitionAndSortWithinPartitions(subtreePartitioner5)

    assert(subtreePartitioner5.getPartition(NodeLabel(33)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(4)))
    assert(subtreePartitioner5.getPartition(NodeLabel(76)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(4)))
    assert(subtreePartitioner5.getPartition(NodeLabel(77)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(4)))
    assert(subtreePartitioner5.getPartition(NodeLabel(82)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner5.getPartition(NodeLabel(83)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner5.getPartition(NodeLabel(92)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner5.getPartition(NodeLabel(56)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner5.getPartition(NodeLabel(57)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner5.getPartition(NodeLabel(58)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(29)))
    assert(subtreePartitioner5.getPartition(NodeLabel(59)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(29)))
    assert(subtreePartitioner5.getPartition(NodeLabel(30)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner5.getPartition(NodeLabel(62)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner5.getPartition(NodeLabel(63)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner5.getPartition(truncWithoutSubtree5.leaves(0)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner5.getPartition(truncWithoutSubtree5.leaves(1)) == subtreePartitioner5.subtreePartitionMap(NodeLabel(28)))

    subtreeRDD5.glom().collect().foreach(arr => {
      val vec = arr.toVector
      var i = 0
      while (i < vec.length) {
        var j = i + 1
        while (j < vec.length) {
          assert(leftRightOrd.compare(vec(i)._1, vec(j)._1) == -1)
          j += 1
        }
        i += 1
      }
    })
  }

  it should "handle sampling duplicates to the partitioner" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val leaves6 = Vector((33,1.0f),(33,1.0f),(33,1.0f),(76,1.0f),(77,1.0f),(82,1.0f),(83,1.0f),(92,1.0f),(56,1.0f),(57,1.0f),(58,1.0f),(59,1.0f),(30,1.0f),(62,1.0f),(63,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val sampleLeaves6 = Vector(33,33,33,76,77,82,83,92,56,57,58,59,30,62,63).map(l => (NodeLabel(l), 1L))
    val maxWeight6 : Float = 3.0f
    val sampleRDD6 = spark.sparkContext.parallelize(sampleLeaves6, 1)
    val subtreePartitioner6 = new SubtreePartitioner(4, sampleRDD6, 15)
    val fullRDD6 = spark.sparkContext.parallelize(sampleLeaves6, 4)
    val subtreeRDD6 = fullRDD6.repartitionAndSortWithinPartitions(subtreePartitioner6)
    val subtrees6 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves6, maxWeight6)

    assert(subtrees6.length == 6)
    assert(subtrees6(0)._1 == NodeLabel(8))
    assert(subtrees6(1)._1 == NodeLabel(9))
    assert(subtrees6(2)._1 == NodeLabel(5))
    assert(subtrees6(3)._1 == NodeLabel(28))
    assert(subtrees6(4)._1 == NodeLabel(29))
    assert(subtrees6(5)._1 == NodeLabel(15))

    assert(subtreePartitioner6.getPartition(NodeLabel(33)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(8)))
    assert(subtreePartitioner6.getPartition(NodeLabel(76)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(9)))
    assert(subtreePartitioner6.getPartition(NodeLabel(77)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(9)))
    assert(subtreePartitioner6.getPartition(NodeLabel(82)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner6.getPartition(NodeLabel(83)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner6.getPartition(NodeLabel(92)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner6.getPartition(NodeLabel(56)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner6.getPartition(NodeLabel(57)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner6.getPartition(NodeLabel(58)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(29)))
    assert(subtreePartitioner6.getPartition(NodeLabel(59)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(29)))
    assert(subtreePartitioner6.getPartition(NodeLabel(30)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner6.getPartition(NodeLabel(62)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner6.getPartition(NodeLabel(63)) == subtreePartitioner6.subtreePartitionMap(NodeLabel(15)))
  }

  it should "throw an error when there are more duplicates than the maximum size of a subtree" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val leaves7 = Vector((33,1.0f),(33,1.0f),(33,1.0f),(33,1.0f),(76,1.0f),(77,1.0f),(82,1.0f),(83,1.0f),(92,1.0f),(56,1.0f),(57,1.0f),(58,1.0f),(59,1.0f),(30,1.0f),(62,1.0f),(63,1.0f)).map(t => (NodeLabel(t._1), t._2))
    val sampleLeaves7 = Vector(33,33,33,33,76,77,82,83,92,56,57,58,59,30,62,63).map(l => (NodeLabel(l), 1L))
    val maxCount7 : Count = 3
    val sampleRDD7 = spark.sparkContext.parallelize(sampleLeaves7, 1)
    val subtreePartitioner7 = new SubtreePartitioner(5, sampleRDD7, 16)
    val fullRDD7 = spark.sparkContext.parallelize(sampleLeaves7, 4)
    val subtreeRDD7 = fullRDD7.repartitionAndSortWithinPartitions(subtreePartitioner7)
    val subtrees7 : Array[(NodeLabel, Float)] = SubtreePartitionerFunctions.maximalWeightSubtreeGeneration(leaves7, maxCount7)
  
    assert(subtrees7.length == 6)
    assert(subtrees7(0)._1 == NodeLabel(8) && subtrees7(0)._2 == 4)
    assert(subtrees7(1)._1 == NodeLabel(9))
    assert(subtrees7(2)._1 == NodeLabel(5))
    assert(subtrees7(3)._1 == NodeLabel(28))
    assert(subtrees7(4)._1 == NodeLabel(29))
    assert(subtrees7(5)._1 == NodeLabel(15))

    assert(subtreePartitioner7.getPartition(NodeLabel(33)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(8)))
    assert(subtreePartitioner7.getPartition(NodeLabel(76)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(9)))
    assert(subtreePartitioner7.getPartition(NodeLabel(77)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(9)))
    assert(subtreePartitioner7.getPartition(NodeLabel(82)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner7.getPartition(NodeLabel(83)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner7.getPartition(NodeLabel(92)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(5)))
    assert(subtreePartitioner7.getPartition(NodeLabel(56)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner7.getPartition(NodeLabel(57)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(28)))
    assert(subtreePartitioner7.getPartition(NodeLabel(58)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(29)))
    assert(subtreePartitioner7.getPartition(NodeLabel(59)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(29)))
    assert(subtreePartitioner7.getPartition(NodeLabel(30)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner7.getPartition(NodeLabel(62)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(15)))
    assert(subtreePartitioner7.getPartition(NodeLabel(63)) == subtreePartitioner7.subtreePartitionMap(NodeLabel(15)))
  }

  "mergeLeavesRDD" should "merge up to the countLimit for a simple case" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val sampleLeaves5 = Vector(16,18,20,22,24,26,28,30).map(l => (NodeLabel(l), 1L))
    val allLeaves5 = Vector(16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31).map(l => (NodeLabel(l), 1L))
    val sampleRDD5 = spark.sparkContext.parallelize(sampleLeaves5, 1)
    val subtreePartitioner5 = new SubtreePartitioner(2, sampleRDD5, 8)
    val fullRDD5 = spark.sparkContext.parallelize(allLeaves5, 4)
    val subtreeRDD5 = fullRDD5.repartitionAndSortWithinPartitions(subtreePartitioner5)

    val countLimit = 4
    val depthLimit = 1

    val rootBox = Rectangle(
      (1 to 1).toVector.map(_ => -10.0),
      (1 to 1).toVector.map(_ => 10.0)
    )
    val tree = widestSideTreeRootedAt(rootBox)

    val merged = mergeLeavesRDD(subtreeRDD5, countLimit, depthLimit).sorted
    assert(merged(0)._1 == NodeLabel(4) && merged(0)._2 == 4)
    assert(merged(1)._1 == NodeLabel(5) && merged(1)._2 == 4)
    assert(merged(2)._1 == NodeLabel(6) && merged(2)._2 == 4)
    assert(merged(3)._1 == NodeLabel(7) && merged(3)._2 == 4)
  }

  it should "handle merging duplicate leafs" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val sampleLeaves5 = Vector(16,16,20,22,24,26,28,30).map(l => (NodeLabel(l), 1L))
    val allLeaves5 = Vector(16,16,16,16,20,21,22,23,24,25,26,27,28,29,30,31).map(l => (NodeLabel(l), 1L))
    val sampleRDD5 = spark.sparkContext.parallelize(sampleLeaves5, 1)
    val subtreePartitioner5 = new SubtreePartitioner(2, sampleRDD5, 8)
    val fullRDD5 = spark.sparkContext.parallelize(allLeaves5, 4)
    val subtreeRDD5 = fullRDD5.repartitionAndSortWithinPartitions(subtreePartitioner5)

    val countLimit = 4
    val depthLimit = 1

    val rootBox = Rectangle(
      (1 to 1).toVector.map(_ => -10.0),
      (1 to 1).toVector.map(_ => 10.0)
    )
    val tree = widestSideTreeRootedAt(rootBox)

    val merged = mergeLeavesRDD(subtreeRDD5, countLimit, depthLimit).sorted
    assert(merged(0)._1 == NodeLabel(4) && merged(0)._2 == 4)
    assert(merged(1)._1 == NodeLabel(5) && merged(1)._2 == 4)
    assert(merged(2)._1 == NodeLabel(6) && merged(2)._2 == 4)
    assert(merged(3)._1 == NodeLabel(7) && merged(3)._2 == 4)
  }

  it should "merge any leftovers if the depthLimit was reached" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    val sampleLeaves5 = Vector(16,18,20,22,24,26,28,30).map(l => (NodeLabel(l), 1L))
    val allLeaves5 = Vector(16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31).map(l => (NodeLabel(l), 1L))
    val sampleRDD5 = spark.sparkContext.parallelize(sampleLeaves5, 1)
    val subtreePartitioner5 = new SubtreePartitioner(2, sampleRDD5, 8)
    val fullRDD5 = spark.sparkContext.parallelize(allLeaves5, 4)
    val subtreeRDD5 = fullRDD5.repartitionAndSortWithinPartitions(subtreePartitioner5)

    val countLimit = 8
    val depthLimit = 2

    val rootBox = Rectangle(
      (1 to 1).toVector.map(_ => -10.0),
      (1 to 1).toVector.map(_ => 10.0)
    )
    val tree = widestSideTreeRootedAt(rootBox)

    val merged = mergeLeavesRDD(subtreeRDD5, countLimit, depthLimit, true).sorted
    assert(merged.length == 2)
    assert(merged(0)._1 == NodeLabel(2) && merged(0)._2 == 8)
    assert(merged(1)._1 == NodeLabel(3) && merged(1)._2 == 8)
  }


  it should "agree with the old mergeLeaves Implementation for uniform data" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd

    var numPartitions = 16
    spark.conf.set("spark.default.parallelism", numPartitions.toString)
    
    val dimensions = 5
    val sizeExp = 3
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-5 
    
    val rootBox = Rectangle(
      (1 to dimensions).toVector.map(_ => 0.0),
      (1 to dimensions).toVector.map(_ => 1.0)
    )

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
    val kInMDE = 10

    val rawTrainRDD = uniformVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 14678)
    val labeledTrainRDD = labelAtDepth(tree, finestResDepth, rawTrainRDD)
    val countedDS = labeledToCountedDS(labeledTrainRDD)
    val countedRDD = countedDS.rdd.repartition(2)

    numPartitions = 2
    
    val partitioner = new SubtreePartitioner(2, countedRDD, 10) /* action 1 (collect) */
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 100
    val subtreeRDD = countedRDD.repartitionAndSortWithinPartitions(partitioner)
    val subtreesMerged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit, true)
    val sortedRDDCollected = subtreesMerged.sorted
    
    val countedTrainDS = countedDS.repartition(2)
    val histogramPath = "./throwaway"
    val mergedDSCollected = mergeLeaves(tree, countedTrainDS, countLimit, stepSize, histogramPath, true).collect.sorted
    
    assert(sortedRDDCollected.length == mergedDSCollected.length)
    for (i <- 0 until sortedRDDCollected.length) {
      assert(sortedRDDCollected(i)._1 == mergedDSCollected(i)._1)
      assert(sortedRDDCollected(i)._2 == mergedDSCollected(i)._2)
    }
  }


  it should "agree with the old mergeLeaves Implementation for normal data" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 5
    val sizeExp = 3
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-4 
    
    val rootBox = Rectangle(
      (1 to dimensions).toVector.map(_ => -10.0),
      (1 to dimensions).toVector.map(_ =>  10.0)
    )

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
    val kInMDE = 10

    val numPartitions = 16

    val rawTrainRDD = uniformVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 49729)
    val labeledTrainRDD = labelAtDepth(tree, finestResDepth, rawTrainRDD)
    val countedDS = labeledToCountedDS(labeledTrainRDD)
    val countedRDD = countedDS.rdd.repartition(2)
    
    val partitioner = new SubtreePartitioner(2, countedRDD, 20) /* action 1 (collect) */
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 50
    val subtreeRDD = countedRDD.repartitionAndSortWithinPartitions(partitioner)
    val subtreesMerged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit, true)
    val sortedRDDCollected = subtreesMerged.sorted
    
    val countedTrainDS = countedDS.repartition(2)
    val histogramPath = "./throwaway"
    val mergedDSCollected = mergeLeaves(tree, countedTrainDS, countLimit, stepSize, histogramPath, true).collect.sorted
    
    assert(sortedRDDCollected.length == mergedDSCollected.length)
    for (i <- 0 until sortedRDDCollected.length) {
      assert(sortedRDDCollected(i)._1 == mergedDSCollected(i)._1)
      assert(sortedRDDCollected(i)._2 == mergedDSCollected(i)._2)
    }
  }
}


class FullRunTests extends FlatSpec with Matchers with BeforeAndAfterAll {

  override protected def beforeAll: Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
    spark.conf.set("spark.default.parallelism", "2")
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  override protected def afterAll: Unit = {
    val spark = getSpark
    spark.stop
  }
  
  val lrOrd : Ordering[NodeLabel] = leftRightOrd

  "Full run of program" should "run without exploding" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 100
    val sizeExp = 5

    val numPartitions = 16
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-5 

    val rawTrainRDD = uniformVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
    val rawTestRDD =  uniformVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)

    var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
    var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectTest)

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
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

    val mde = getMDE(
      Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap)), 
      countedTest, 
      kInMDE, 
      true
    )
  }
}
