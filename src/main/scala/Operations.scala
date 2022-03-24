package co.wiklund.disthist

import scala.reflect.ClassTag

import TruncationFunctions._
import NodeLabelFunctions._
import TruncationOperations._
import LeafMapFunctions._
import LeafMapOperations._
import SpatialTreeFunctions._
import HistogramOperations._
import Types._
import org.apache.spark.mllib.linalg.Vectors

object TruncationOperations {
  private object DropHead extends Enumeration {
    type DropHead = Value
    val L1, L2, Both = Value
  }
  import DropHead._

  private def getDrop(n1: NodeLabel, n2: NodeLabel): (NodeLabel, DropHead) = {
    if (n1 == n2 || isAncestorOf(n2, n1))
      (n1, Both)
    else if (isAncestorOf(n1, n2))
      (n2, Both)
    else if (isLeftOf(n1, n2))
      (n1, L1)
    else
      (n2, L2)
  }

  @scala.annotation.tailrec
  private def unionHelper(
    acc: Vector[NodeLabel],
    l1: Vector[NodeLabel], 
    l2: Vector[NodeLabel]
  ): Vector[NodeLabel] = {
    if (l1.isEmpty)
      acc ++ l2
    else if (l2.isEmpty)
      acc ++ l1
    else {
      val n1 = l1.head
      val n2 = l2.head
      val (next, drop) = getDrop(n1, n2)
      unionHelper(
        acc :+ next, 
        if (drop != L2) l1.tail else l1,
        if (drop != L1) l2.tail else l2
      )
    }
  }

  /**
    * The leaves of the tree resulting from `RPUnion(t1, t2)`
    */
  def rpUnion(t1: Truncation, t2: Truncation): Truncation = {
    Truncation(unionHelper(Vector.empty, t1.leaves, t2.leaves))
  }
   
  /**
    * WARNING: Does not check that `descendent` has an ancestor in the tree.
    *
    * @param trunc
    * @param descendent
    * @return
    */
  def findAncestor(trunc: Truncation, descendent: NodeLabel): NodeLabel = {
    trunc.leaves.find(isAncestorOf(_, descendent)).get
  }
}

object LeafMapOperations {
  def mrpTransform[A, B:ClassTag](leafMap: LeafMap[A], f: A => B): LeafMap[B] = {
    leafMap.copy(vals = leafMap.vals.map(f))
  }
  
  def mrpOperate[A:ClassTag](leafMap1: LeafMap[A], leafMap2: LeafMap[A], op: (A, A) => A): LeafMap[A] = {
    val t1 = leafMap1.truncation
    val t2 = leafMap2.truncation
    val t1nodes = t1.leaves
    val t2nodes = t2.leaves
    val commonNodes = t1nodes intersect t2nodes
    val t1only = Truncation(t1.leaves diff t2.leaves)
    val t2only = Truncation(t2.leaves diff t1.leaves)
    val t1missing = t1only.leaves.flatMap((node: NodeLabel) => t2only.viewSubtree(node, 0)).toVector.distinct // missing from t1
    val t2missing = t2only.leaves.flatMap((node: NodeLabel) => t1only.viewSubtree(node, 0)).toVector.distinct // missing from t2

    val unionedTree = (t1missing union commonNodes.toVector union t2missing).toSet

    val ancestorsInT1 = t1only.leaves diff t2missing
    val ancestorsInT2 = t2only.leaves diff t1missing

    def getUnionMap(
      unionedTree: Set[NodeLabel], 
      leafMap: Map[NodeLabel, A], 
      missingNodes: Vector[NodeLabel], 
      ancestorsInMissing: Vector[NodeLabel]
    ): Map[NodeLabel, A] = {
      var i = 0
      val ancestorsOfMissing = missingNodes.tail.scanLeft(ancestorsInMissing.head)(
        (anc, node) => 
          if (isAncestorOf(anc, node)) 
            anc 
          else {
            i += 1
            ancestorsInMissing(i)
          })

      val missingVals = ancestorsOfMissing.map(leafMap(_))
      (leafMap ++ missingNodes.zip(missingVals)).filterKeys(unionedTree.contains(_))
    }
   
    val updatedMap1 = getUnionMap(unionedTree, leafMap1.toMap, t1missing, ancestorsInT1)
    val updatedMap2 = getUnionMap(unionedTree, leafMap2.toMap, t2missing, ancestorsInT2)

    val operatedLeafMap = (updatedMap1.toVector ++ updatedMap2.toVector).groupBy(_._1).mapValues(_.map(_._2).reduce(op))
    
    fromNodeLabelMap(operatedLeafMap)
  }
}

case class DensityHistogram(tree: SpatialTree, densityMap: LeafMap[(Probability, Volume)]) {
  def density(v: MLVector): Probability = 
    densityMap.query(tree.descendBox(v))._2.getOrElse((0.0, 0.0))._1

  def normalize: DensityHistogram = {
    val probSum = densityMap.vals.map{ case (prob, _) => 
      prob 
    }.sum
    val normVals = densityMap.vals.map{ case (prob, vol) =>
      (prob / (vol * probSum), vol)
    }
    copy(densityMap = densityMap.copy(vals = normVals))
  }
}

object HistogramOperations {
  def toDensityHistogram(hist: Histogram): DensityHistogram = {
    val counts = hist.counts
    val tree = hist.tree
    val totalCount = hist.totalCount

    val densitiesWithVolumes = counts.toIterable.map {
      case (lab, c) => 
        val vol = tree.volumeAt(lab)
        (lab, (c/(totalCount * vol), vol))
    }.toMap

    DensityHistogram(tree, fromNodeLabelMap(densitiesWithVolumes))
  }
  
  /**
    * Removes axes from a rectangle
    *
    * @param rec
    * @param axesToKeep
    * @return
    */
  def marginalizeRectangle(rec: Rectangle, axesToKeep: Vector[Axis]): Rectangle = {
    Rectangle(axesToKeep map rec.low, axesToKeep map rec.high)
  }

  /**
    * The bit-value of the label as a vector
    * 
    * ex. 14 = b1110, so
    * getSplitDirections(NodeLabel(14)) = Vector(1,1,1,0)
    *
    * @param lab
    * @return
    */
  def getSplitDirections(lab: NodeLabel): Vector[Int] = {
    (0 to lab.depth - 1).toVector.map(lab.lab.testBit).map(if (_) 1 else 0)
  }

  def marginalize(hist: Histogram, axesToKeep: Vector[Axis]): DensityHistogram = {
    marginalize(toDensityHistogram(hist), axesToKeep)
  }

  def marginalize(densHist: DensityHistogram, axesToKeep: Vector[Axis]): DensityHistogram = {
    val densTree = densHist.tree
    val densAndVols = densHist.densityMap

    val splits = densTree.splits

    val marginalized = densAndVols.toIterable.groupBy{ case (lab, _) => 
      // group by boxes with unwanted axes removed
      marginalizeRectangle(densTree.cellAt(lab), axesToKeep)
    }.map{ case (newRec, densities) => 
      val newVol = newRec.volume
      // aggregate by sum of densities in the new box
      val newDensity = densities.map { case (_, (dens, vol)) =>
        dens * vol / newVol
      }.sum 
      // Find NodeLabel corresponding to new box
      val newLabel = {
        val oldLab = densities.head._1
        val axisSplits = splits.take(oldLab.depth).reverse
        val splitDirections = getSplitDirections(oldLab)
        val newDirections = axisSplits.zip(splitDirections).filter{ case (axis, _) =>
          axesToKeep.contains(axis)
        }.map{ case (_, direction) => direction }
        NodeLabel(newDirections.reverse.foldLeft(BigInt(1)){ case (labAcc, i) => (labAcc << 1) + i })
      } 
      (newRec, (newLabel, newDensity, newVol))
    }.map{ case(rec, (lab, dens, vol)) => (lab, (dens: Probability, vol)) }

    val margTree = widestSideTreeRootedAt(marginalizeRectangle(densTree.rootCell, axesToKeep))

    // Find which boxes overlap with their descendants
    val ancToDescs = marginalized.keys.toVector
      .flatMap(node => 
        (1 to node.depth-1).reverse.map(i => (node, node.truncate(i)))
      ).groupBy{ case (desc, anc) => anc }
      .mapValues(vec => vec.map{ case (desc, anc) => desc })

    // Add missing siblings of descendants
    val ancToDescsWithMissing = ancToDescs.mapValues(descs =>
      descs.union(descs.map(_.sibling)).distinct
    )

    // Map descendents to the sum of densities of ancestor boxes
    val descsWithDens = ancToDescsWithMissing.toVector
      .flatMap{ case (anc, descs) => 
        descs.map(desc => (desc, marginalized.getOrElse(anc, (0.0, 0.0))._1)) 
      }.groupBy{ case (lab, dens) => lab }
      .mapValues( labAndDens => 
        labAndDens.map{ case (lab, dens) => dens }.sum
      )

    // Add the densities from ancestor boxes to the descendants,
    // and remove all nodes that are not leaves.
    val margMap = descsWithDens.keys.foldLeft(marginalized){
      case (mapAcc, descLab) =>
        val oldValue = mapAcc.getOrElse(descLab, (0.0, margTree.volumeAt(descLab)))
        val newValue = (oldValue._1 + descsWithDens(descLab), oldValue._2)
        mapAcc.updated(descLab, newValue)
    }.filter{ case (lab, _) => !ancToDescs.keySet.contains(lab) }

    DensityHistogram(margTree, fromNodeLabelMap(margMap))
  }

  def slice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoints: Vector[Double]): DensityHistogram = {
    
    val mlSlicePoints = Vectors.dense(slicePoints.toArray)
    val sliceBoxes = densHist.densityMap.truncation.leaves.map( node => 
      node -> marginalizeRectangle(densHist.tree.cellAt(node), sliceAxes)
    ).toMap.filter{ case (lab, rec) => 
      rec.contains(mlSlicePoints)
    }

    val splits = densHist.tree.splits
    val nonSliceAxes = ((0 to densHist.tree.dimension - 1).toSet -- sliceAxes).toVector

    val newLabels = sliceBoxes.keys.toVector.map{ lab => 
      val axisSplits = splits.take(lab.depth).reverse
      val splitDirections = getSplitDirections(lab)

      val newDirections = axisSplits.zip(splitDirections)
        .filter{ case (axis, _) =>
          nonSliceAxes.contains(axis)
        }.map{ case (_, direction) => direction }

      val newLab = newDirections.reverse
        .foldLeft(BigInt(1)){ case (labAcc, i) => 
          (labAcc << 1) + i 
        }

      lab -> NodeLabel(newLab)
    }

    val densMap = densHist.densityMap.toMap
    val slicedRootBox = marginalizeRectangle(densHist.tree.rootCell, nonSliceAxes)
    val slicedTree = widestSideTreeRootedAt(slicedRootBox)

    val slicedNodeMap = newLabels.map{ case (oldLab, newLab) =>
      newLab -> (densMap(oldLab)._1, slicedTree.volumeAt(newLab))
    }.toMap

    val newTrunc = fromLeafSet(slicedNodeMap.keySet)
    val missingNodes = newTrunc.minimalCompletion.leaves.toSet -- newTrunc.leaves.toSet

    val slicedNodeMapWithMissing = slicedNodeMap ++ missingNodes.map( node => node -> (0.0, slicedTree.volumeAt(node)) )

    DensityHistogram(slicedTree, fromNodeLabelMap(slicedNodeMapWithMissing))
  }

  def toCollatedHistogram[K](hist: Histogram, key: K): CollatedHistogram[K] = {
    toCollatedHistogram(toDensityHistogram(hist), key)
  }

  def toCollatedHistogram[K](hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    val mappedDensities = hist.densityMap.vals.map(v => Map(key -> v))
    CollatedHistogram(hist.tree, hist.densityMap.copy(vals = mappedDensities))
  }
}

case class CollatedHistogram[K](tree: SpatialTree, densities: LeafMap[Map[K, (Probability, Volume)]]) {
  private type MapType = Map[K, (Probability, Volume)]

  protected val keySet = densities.vals.head.keySet
  private def collatorOp(kv1: MapType, kv2: MapType): MapType = {
    val all = kv1 ++ kv2
    val minVol = all.values.map{ case (_, vol) => vol }.min
    all.mapValues{ case (dens, vol) => (dens * minVol / vol, minVol)}
  }

  def collate(hist: Histogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key))
  }

  def collate(hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key))
  }

  def collate(hist: CollatedHistogram[K]): CollatedHistogram[K] = {
    if (keySet.intersect(hist.keySet).nonEmpty)
      throw new IllegalArgumentException(s"keySets are not allowed to intersect. The common keys are ${keySet intersect hist.keySet}.")

    if (tree != hist.tree)
      throw new IllegalArgumentException("Collated histograms must have the same root box.")

    CollatedHistogram(tree, mrpOperate(densities, hist.densities, collatorOp))
  }
}