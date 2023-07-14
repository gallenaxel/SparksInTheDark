/**************************************************************************
 * Copyright 2022 Johannes Graner
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/

package co.wiklund.disthist

import Types._
import LeafMapFunctions._
import SpatialTreeFunctions._
import NodeLabelFunctions._
import TruncationFunctions._
import HistogramFunctions._
import GslRngHandle._

import org.apache.spark.mllib.linalg.Vectors

case class DensityHistogram(tree: SpatialTree, densityMap: LeafMap[(Probability, Volume)]) {
  def density(v: MLVector): Probability = {
    val point = v.toArray
    for (i <- 0 until point.length) {
      if (point(i) < tree.rootCell.low(i) || point(i) > tree.rootCell.high(i)) {
        return 0.0
      }
    }

    densityMap.query(tree.descendBox(v))._2.getOrElse((0.0, 0.0))._1
  }

  def normalize: DensityHistogram = {
    val integral = densityMap.vals.map{ case (dens, vol) => 
      dens * vol
    }.sum
    val normVals = densityMap.vals.map{ case (dens, vol) =>
      (dens / integral, vol)
    }
    copy(densityMap = densityMap.copy(vals = normVals))
    DensityHistogram(this.tree, this.densityMap.copy(vals = normVals))
  }

  /**
   * Sample from the distribution defined by the density. Note that the given handle must be a handle of a valid (allocated)
   * GNU Scientific Library random number generator (gsl_rng).
   *
   * @param handle - Initialized GslRngHandle to a random number generator (gsl_rng) C struct
   * @param sampleSize - Wanted size
   * @return A Sample of size sampleSize from the density histogram's distribution
   */
  def sample(handle : GslRngHandle, sampleSize : Int): Array[Array[Double]] = {
    var sampleBuf : Array[Array[Double]] = new Array(sampleSize)
     
  }
}

object HistogramFunctions {
  def toDensityHistogram(hist: Histogram): DensityHistogram = {
    val counts = hist.counts
    val tree = hist.tree
    val totalCount = hist.totalCount
    val totalVolume = tree.rootCell.volume

    val densitiesWithVolumes = counts.toIterable.map {
      case (node, c) => 
        val vol = tree.volumeAt(node)
        (node, (c / (totalCount * vol), vol))
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

  private def collatorOp(allKeys: Set[K])(kv1: MapType, kv2: MapType): MapType = {
    if (kv1.isEmpty)
      kv2 ++ ((allKeys -- kv2.keySet).map(key => key -> (0.0, 0.0)))
    else if (kv2.isEmpty)
      kv1 ++ ((allKeys -- kv1.keySet).map(key => key -> (0.0, 0.0)))
    else 
      kv1 ++ kv2
  }

  def collate(hist: Histogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key))
  }

  def collateNested(hist: Histogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key), true)
  }

  def collate(hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key))
  }

  def collateNested(hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key), true)
  }

  def collateNested(hist: CollatedHistogram[K]): CollatedHistogram[K] = {
    collate(hist, true)
  }

  def collate(hist: CollatedHistogram[K], nested: Boolean = false): CollatedHistogram[K] = {
    if (keySet.intersect(hist.keySet).nonEmpty)
      throw new IllegalArgumentException(s"keySets are not allowed to intersect. The common keys are ${keySet intersect hist.keySet}.")

    if (tree != hist.tree)
      throw new IllegalArgumentException("Collated histograms must have the same root box.")

    val allKeys = keySet union hist.keySet
    val base: MapType = Map.empty

    val collatedDensityMap = mrpOperate(densities, hist.densities, collatorOp(allKeys), base, nested)
    CollatedHistogram(
      tree, 
      collatedDensityMap.copy(
        vals = collatedDensityMap.toIterable.toVector.map{ case (node, densMap) =>
          val correctVol = tree.volumeAt(node)
          densMap.mapValues{ case (dens, vol) => (dens, correctVol)
        }  
      })
    )
  }
}
