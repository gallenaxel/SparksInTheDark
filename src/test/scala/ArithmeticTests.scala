import scala.language.postfixOps

import org.apache.spark.mllib.linalg.{ Vector => MLVector, _ }
import scala.math.abs

import org.scalatest.{ path => testPath, _ }
import org.scalactic.TolerantNumerics
import org.scalactic.TripleEquals._

import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.LeafMapFunctions._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.TruncationFunctions._

class ArithmeticTests extends FlatSpec with Matchers {

  val tn: Int => NodeLabel = NodeLabel(_)

  "rpUnion" should "generate only the leaves of the unioned tree" in {
    val nodes1 = Vector(8,9,5,3).map(NodeLabel(_))
    val trunc1 = Truncation(nodes1)
    val nodes2 = Vector(4,10,11,3).map(NodeLabel(_))
    val trunc2 = Truncation(nodes2)
    val unioned = rpUnion(trunc1, trunc2)
    val expectedLeaves = Vector(8,9,10,11,3)
    assert(unioned.leaves.map(_.lab) === expectedLeaves)
  }

  it should "work with several levels of ancestry" in {
    val t1 = rootTruncation
    val t2 = Truncation(Vector(5,6).map(tn))
    val union = rpUnion(t1, t2)
    val expectedLeaves = Vector(4,5,6,7)
    assert(union.leaves.map(_.lab) === expectedLeaves) 
  }

  it should "work with sparse trees" in {
    val t1 = Truncation(Vector(4,5).map(NodeLabel(_)))
    val t2 = Truncation(Vector(16,6,15).map(NodeLabel(_)))
    val unioned = rpUnion(t1, t2)
    val expected = Truncation(Vector(16,17,9,5,6,15).map(NodeLabel(_)))
    assert(unioned === expected)
  }

  "mrpTransform" should "be a pointwise transformation" in {
    val trunc = Truncation(Vector(8,9,10,11,3).map(NodeLabel(_)))
    val vals = Vector(1,2,3,4,5)
    val leafMap = LeafMap(trunc, vals)
    assert(mrpTransform(leafMap, (x: Int) => x * 2).toMap.apply(NodeLabel(10)) === 6)
  }

  "mrpOperate" should "generate the correct tree" in {
    val nodes1 = Vector(8,9,5,3).map(NodeLabel(_))
    val trunc1 = Truncation(nodes1)
    val vals1 = Vector(0.1, 0.2, 0.3, 0.4)
    val leafMap1 = LeafMap(trunc1, vals1)

    val nodes2 = Vector(4,10,11,3).map(NodeLabel(_))
    val trunc2 = Truncation(nodes2)
    val vals2 = Vector(0.4, 0.3, 0.2, 0.1)
    val leafMap2 = LeafMap(trunc2, vals2)

    val op = (x: Double, y: Double) => (x + y) / 2
    val base = 0.0

    val expectedTrunc = Truncation(Vector(8,9,10,11,3).map(NodeLabel(_)))
    val expectedVals = Vector(0.25, 0.3, 0.3, 0.25, 0.25)

    val result = mrpOperate(leafMap1, leafMap2, op, base)
    val result2 = mrpOperate(leafMap2, leafMap1, op, base)
    assert(result === result2)
    assert(result.truncation === expectedTrunc)
    assert(result.vals.zip(expectedVals).forall{ case (x,y) => math.abs(x - y) < 1e-10})
  }

  it should "work with several levels of ancestry" in {
    val t1 = rootTruncation
    val v1 = Vector(1)
    val leafMap1 = LeafMap(t1, v1)
    val t2 = Truncation(Vector(5,6).map(tn))
    val v2 = Vector(1,2)
    val leafMap2 = LeafMap(t2, v2)
    val op = (x: Int, y: Int) => x + y

    val expectedLeaves = Vector(4,5,6,7)
    val expectedVals = Vector(1,2,3,1)

    val result = mrpOperate(leafMap1, leafMap2, op, 0)
    assert(result.leaves.map(_.lab) === expectedLeaves) 
    assert(result.vals === expectedVals)
  }

  it should "work with sparse trees" in {
    val nodes1 = Vector(16,3).map(NodeLabel(_))
    val trunc1 = Truncation(nodes1)
    val vals1 = Vector(0.1, 0.3)
    val leafMap1 = LeafMap(trunc1, vals1)

    val nodes2 = Vector(4,11,3).map(NodeLabel(_))
    val trunc2 = Truncation(nodes2)
    val vals2 = Vector(0.4, 0.2, 0.1)
    val leafMap2 = LeafMap(trunc2, vals2)

    val op = (x: Double, y: Double) => (x + y) / 2
    val base = 0.0

    val expectedTrunc = Truncation(Vector(16,17,9,11,3).map(NodeLabel(_)))
    val expectedVals = Vector(0.25, 0.2, 0.2, 0.1, 0.2)

    val result = mrpOperate(leafMap1, leafMap2, op, base)
    val result2 = mrpOperate(leafMap2, leafMap1, op, base)
    assert(result === result2)
    assert(result.truncation === expectedTrunc)
    assert(result.vals === expectedVals)
    // assert(result.vals.zip(expectedVals).forall{ case (x,y) => math.abs(x - y) < 1e-10})
  }
}

class OperationTests extends FlatSpec with Matchers with BeforeAndAfterAll {

  private var margHist: DensityHistogram = null
  private var denseHist: DensityHistogram = null
  private val tn: Int => NodeLabel = NodeLabel(_)
  override protected def beforeAll(): Unit = {
    val rootBox = Rectangle(Vector(0.0, 0.0), Vector(1.0, 1.0))
    val axesToKeep = Vector(0)
    val tree = widestSideTreeRootedAt(rootBox)
    val nodes = Vector(4, 10, 11, 6, 14, 15).map(NodeLabel(_))
    val counts: Vector[Count] = Vector(2,1,2,1,2,2)
    val totalCount = counts.sum
    val leafMap = fromNodeLabelMap(nodes.zip(counts).toMap)
    val hist = Histogram(tree, totalCount, leafMap)
    margHist = marginalize(hist, axesToKeep)
    
    denseHist = toDensityHistogram(hist)
  }

  "marginalize" should "give proper density" in {
    val error = abs(margHist.densityMap.vals.map{ case (dens, vol) => dens * vol }.sum - 1)
    assert(error < 1e-10 )
  }

  it should "have the correct truncation" in {
    val expectedLeaves = Vector(4,5,6,7).map(NodeLabel(_))
    assert(margHist.densityMap.truncation === Truncation(expectedLeaves))
  }
  
  it should "have the correct volumes" in {
    val margVols = margHist.densityMap.vals.map(_._2)
    val expectedVols = Vector(0.25, 0.25, 0.25, 0.25)
    assert(margVols === expectedVols)
  }

  it should "have the correct densities" in {
    val expectedLeaves = Vector(4,5,6,7).map(NodeLabel(_))
    val expectedDensities = Vector(0.8, 1.2, 1.0, 1.0)
    val expectedLeafMap = fromNodeLabelMap(expectedLeaves.zip(expectedDensities).toMap)
    
    val densDiffs = margHist.densityMap.vals.map(_._1).zip(expectedDensities).map{ case (a, b) => abs(a - b) }

    densDiffs.sum should be < 1e-10
  }

  it should "work with sparse trees" in {
    val rootBox = Rectangle(Vector(0.0, 0.0), Vector(1.0, 1.0))
    val axesToKeep = Vector(0)
    val tree = widestSideTreeRootedAt(rootBox)
    val nodes = Vector(2,12,59,61).map(NodeLabel(_))
    val counts: Vector[Count] = Vector(1,1,1,1)
    val totalCount = counts.sum
    val leafMap = fromNodeLabelMap(nodes.zip(counts).toMap)
    val hist = Histogram(tree, totalCount, leafMap)
    val margHist = marginalize(hist, axesToKeep)

    val expectedRootBox = Rectangle(Vector(0.0), Vector(1.0))
    val expectedLeaves = Vector(2,12,13,15).map(NodeLabel(_))
    val expectedVols = Vector(0.5, 0.125, 0.125, 0.125)
    val expectedDens = Vector(0.5, 1.0, 3.0, 2.0)

    margHist.tree.rootCell shouldEqual expectedRootBox
    margHist.densityMap.leaves shouldEqual expectedLeaves
    margHist.densityMap.vals shouldEqual expectedDens.zip(expectedVols)
  }

  it should "work with non-unit-cube domain" in {
    val rootBox = Rectangle(Vector(0.0, 0.0), Vector(2.0, 1.0))
    val axesToKeep = Vector(0)
    val hist = Histogram(
      widestSideTreeRootedAt(rootBox),
      6L,
      LeafMap(Truncation(Vector(8,19,3) map tn), Vector(2L, 3L, 1L))
    )

    val densHist = toDensityHistogram(hist)
    (densHist.densityMap.vals.map{ case (dens, vol) => dens * vol }.sum - 1.0) should be < 1e-10

    val margHist = marginalize(densHist, axesToKeep)

    val expectedRootBox = Rectangle(Vector(0.0), Vector(2.0))
    val expectedLeaves = Vector(8,9,3) map tn
    val expectedVols = Vector(0.25, 0.25, 1.0)
    val expectedDens = Vector(4.0, 16.0, 1.0).map(_ / 6)

    val densityError = margHist.densityMap.vals.map{ case (dens, vol) => dens * vol }.sum - 1.0

    margHist.tree.rootCell shouldEqual expectedRootBox
    margHist.densityMap.leaves shouldEqual expectedLeaves
    margHist.densityMap.vals shouldEqual expectedDens.zip(expectedVols)
    densityError should be < 1e-10
  }

  "slice" should "have the correct truncation" in {
    val sliceAxes = Vector(0)
    val slicePoints = Vectors.dense(0.6)
    val sliceHist = slice(denseHist, sliceAxes, slicePoints)

    val expectedTrunc = Truncation(Vector(2,3).map(tn))
    sliceHist.densityMap.truncation shouldEqual expectedTrunc
  }

  it should "have the correct volumes" in {
    val sliceAxes = Vector(0)
    val slicePoints = Vectors.dense(0.6)
    val sliceHist = slice(denseHist, sliceAxes, slicePoints)

    val expectedVols = Vector(0.5, 0.5)
    val sliceVols = sliceHist.densityMap.vals.map{ case (_, vol) => vol }
    sliceVols shouldEqual expectedVols
  }

  it should "have the correct densities" in {
    val sliceAxes = Vector(0)
    val slicePoints = Vectors.dense(0.6)
    val sliceHist = slice(denseHist, sliceAxes, slicePoints)

    val expectedDensities = Vector(0.4, 1.6)
    val sliceDensities = sliceHist.densityMap.vals.map{ case (dens, _) => dens }
    sliceDensities shouldEqual expectedDensities
  }

  it should "work with sparse trees" in {
    val leaves = Vector(2,12,59,61).map(tn)
    val vals: Vector[Count] = Vector(1,1,1,1)
    val totalCount = vals.sum
    val rootBox = Rectangle(Vector(0.0,0.0), Vector(1.0,1.0))
    val tree = widestSideTreeRootedAt(rootBox)
    val hist = Histogram(tree, totalCount, fromNodeLabelMap((leaves zip vals).toMap))
    val densHist = toDensityHistogram(hist)

    val sliceAxes = Vector(0)
    val slicePoints = Vectors.dense(0.7)
    val sliceHist = slice(densHist, sliceAxes, slicePoints)

    val expectedRootBox = Rectangle(Vector(0.0), Vector(1.0))
    val expectedLeaves = Vector(2,7).map(tn)
    val expectedVols = Vector(0.5, 0.25)
    val expectedDens = Vector(2.0, 8.0)

    sliceHist.tree.rootCell shouldEqual expectedRootBox
    sliceHist.densityMap.leaves shouldEqual expectedLeaves
    sliceHist.densityMap.vals shouldEqual (expectedDens zip expectedVols)
  }

  {
    val rootBox = Rectangle(Vector(0.0, 0.0), Vector(2.0, 1.0))
    val hist = Histogram(
      widestSideTreeRootedAt(rootBox),
      6L,
      LeafMap(Truncation(Vector(8,19,3) map tn), Vector(2L, 3L, 1L))
    )

    val densHist = toDensityHistogram(hist)

    "toDensityHistogram" should "yield density" in {
      val totalVol = densHist.tree.volumeTotal
      val error = densHist.densityMap.vals.map{ case (dens, vol) => dens * vol }.sum - 1.0
      abs(error) should be < 1e-10
    }

    "normalize" should "not change normalized histogram" in {
      val normalized = densHist.normalize

      val absDiff: ((Probability, Probability)) => Probability = { case (x, y) => abs(x - y) }
      val densDiff = (normalized.densityMap.vals.map(_._1) zip densHist.densityMap.vals.map(_._1)).map(absDiff).sum
      
      normalized.tree shouldEqual densHist.tree
      normalized.densityMap.truncation shouldEqual densHist.densityMap.truncation
      normalized.densityMap.vals.map(_._2) shouldEqual densHist.densityMap.vals.map(_._2)
      densDiff should be < 1e-10
    }
  }

  {
    val rootBox = Rectangle(Vector(0.0, 0.0), Vector(1.0, 1.0))
    val hist1 = Histogram(
      widestSideTreeRootedAt(rootBox),
      10L,
      LeafMap(Truncation(Vector(8,3) map tn), Vector(3L, 7L))
    )
    val hist2 = Histogram(
      widestSideTreeRootedAt(rootBox),
      4L,
      LeafMap(Truncation(Vector(4,31) map tn), Vector(2L, 2L))
    )

    val collHist = toCollatedHistogram(hist1, "1").collate(hist2, "2")

    val expectedLeaves = Vector(8,9,6,14,30,31) map tn
    val expectedVols = Vector(0.125, 0.125, 0.25, 0.125, 0.0625, 0.0625)
    val expectedDens1 = Vector(2.4, 0.0, 1.4, 1.4, 1.4, 1.4)
    val expectedDens2 = Vector(2.0, 2.0, 0.0, 0.0, 0.0, 8.0)
    val expectedKeyset = Set("1", "2")

    "CollatedHistogram" should "have the correct keys" in {
      collHist.keySet shouldEqual expectedKeyset
    }

    it should "have the correct leaves" in {
      collHist.densities.leaves shouldEqual expectedLeaves
    }

    it should "have the correct volumes" in {
      val collVols = collHist.densities.vals.map(densMap => densMap.values.map{ case (_, vol) => vol }.toVector.distinct)
      collVols shouldEqual expectedVols.map(Vector(_))
    }

    it should "have the correct densities" in {
      val collDenses1 = collHist.densities.vals.map(densMap => densMap("1")._1)
      val collDenses2 = collHist.densities.vals.map(densMap => densMap("2")._1)
      
      collDenses1 shouldEqual expectedDens1
      collDenses2 shouldEqual expectedDens2
    }
  }
}