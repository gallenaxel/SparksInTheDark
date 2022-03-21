package co.wiklund.disthist

import scala.reflect.ClassTag

import TruncationFunctions._
import NodeLabelFunctions._
import TruncationOperations._
import LeafMapFunctions._
import LeafMapOperations._
import Types._

object TruncationOperations {
  /**
    * The leaves of the tree resulting from `RPUnion(t1, t2)`
    */
  def rpUnion(t1: Truncation, t2: Truncation): Truncation = {
    val t1nodes = t1.leaves.toSet
    val t2nodes = t2.leaves.toSet
    val commonNodes = t1nodes intersect t2nodes
    val t1only = Truncation(t1.leaves diff t2.leaves)
    val t2only = Truncation(t2.leaves diff t1.leaves)
    val t1missing = t1only.leaves.toSet.flatMap((node: NodeLabel) => t2only.viewSubtree(node, 0))
    val t2missing = t2only.leaves.toSet.flatMap((node: NodeLabel) => t1only.viewSubtree(node, 0))
    fromLeafSet(t1missing union commonNodes union t2missing)
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

    def getUnionMap(unionedTree: Set[NodeLabel], leafMap: Map[NodeLabel, A], missingNodes: Vector[NodeLabel], ancestorsInMissing: Vector[NodeLabel]): Map[NodeLabel, A] = {
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

  def marginalize(hist: Histogram, keptAxes: Vector[Axis]): Histogram = {
    val leaves = hist.counts.truncation.leaves
    

    ???
  }
}