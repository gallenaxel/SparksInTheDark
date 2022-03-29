package co.wiklund.disthist

import Types._
import LeafMapFunctions._
import SpatialTreeFunctions._
import NodeLabelFunctions._
import TruncationFunctions._
import HistogramFunctions._

import org.apache.spark.mllib.linalg.Vectors

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

object HistogramFunctions {
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
    (0 to lab.depth - 1).toVector.map(i => 
      if (lab.lab.testBit(i)) 1 else 0
    )
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

    // find the leaves in the marginalized tree,
    // and the ancestors grouped by depth
    val (margLeaves, margAncesAtDepth) = marginalized.keySet
      .groupBy(_.depth).toVector.sortBy(-_._1)
      .foldLeft((Set.empty[NodeLabel], Map.empty[Depth, Set[NodeLabel]])){
        case ((leafAcc, ancesAtDepths), (depth, nodesAtDepth)) =>
          val leavesAtDepth = leafAcc.map(_.truncate(depth))
          val newLeaves = nodesAtDepth -- leavesAtDepth
          val ancesAtDepth = depth -> (nodesAtDepth intersect leavesAtDepth)

          (leafAcc ++ newLeaves, ancesAtDepths + ancesAtDepth)
      }

    // find the missing leaves in the marginalized tree,
    // i.e. the ones which are not already leaves, but
    // have an ancestor in the keys of marginalization.
    val ancToDescsWithMissing = margAncesAtDepth.flatMap{
      case (depth, ancesAtDepth) =>

        // Finds the label for `node` if the ancestor `root` was the root node.
        def rootAtNode(root: NodeLabel, node: NodeLabel): NodeLabel = 
          NodeLabel(
            ( node.lab - 
              (root.lab << (node.depth - root.depth))
            ).setBit(node.depth - root.depth)
          )

        // Finds the label for `node` if it's tree
        // was grafted to `root`.
        // Inversion of rootAtNode.
        def descendantFromRoot(root: NodeLabel, node: NodeLabel): NodeLabel =
          NodeLabel(
            (root.lab << (node.depth)) + 
            (node.lab.clearBit(node.depth))
          )

        // descendants to ancestors at the current depth
        val descs = margLeaves.filter(node => 
          ancesAtDepth.contains(node.truncate(depth))
        )

        // grouping descendants by their ancestor
        val ancToDescs = ancesAtDepth.map(anc => 
          anc -> descs.filter(desc => 
            isAncestorOf(anc, desc)
          )
        )

        // filling in the missing nodes by finding the minimal
        // completion of the subtree.
        ancToDescs.map{ case (anc, descs) =>
          val rootedDescs = descs.map( desc => 
            rootAtNode(anc, desc)
          )
          val rootedWithMissing = fromLeafSet(rootedDescs).minimalCompletion.leaves
          val descsWithMissing = rootedWithMissing.map(rootedDesc => 
            descendantFromRoot(anc, rootedDesc)
          )
          anc -> descsWithMissing
        }
    }

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
    }.filter{ case (lab, _) => !ancToDescsWithMissing.keySet.contains(lab) }

    DensityHistogram(margTree, fromNodeLabelMap(margMap))
  }

  def slice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoints: Vector[Double]): DensityHistogram = {
    slice(densHist, sliceAxes, Vectors.dense(slicePoints.toArray))
  }

  def slice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoints: MLVector): DensityHistogram = {
    
    val sliceBoxes = densHist.densityMap.leaves.map( node => 
      node -> marginalizeRectangle(densHist.tree.cellAt(node), sliceAxes)
    ).toMap.filter{ case (lab, rec) => 
      rec.contains(slicePoints)
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

    DensityHistogram(slicedTree, fromNodeLabelMap(slicedNodeMap))
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

  val keySet = densities.vals.head.keySet

  private def collatorOp(kv1: MapType, kv2: MapType): MapType = {
    val all = kv1 ++ kv2
    val minVol = all.values.map{ case (_, vol) => vol }.min
    all.mapValues{ case (dens, vol) => (dens, minVol)}
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

    def addSiblings(density: LeafMap[MapType]): LeafMap[MapType] = {
      val leaves = density.leaves
      val missingSiblings = leaves.map(_.sibling).toSet -- leaves
      val keys = density.vals.head.keySet
      val missingMap = missingSiblings.map{ node => 
        val vol = tree.volumeAt(node)
        node -> keys.map(_ -> (0.0, vol)).toMap
      }
      
      val newMap = density.toMap ++ missingMap
      fromNodeLabelMap(newMap)
    }

    CollatedHistogram(tree, mrpOperate(addSiblings(densities), addSiblings(hist.densities), collatorOp, Map.empty))
  }
}