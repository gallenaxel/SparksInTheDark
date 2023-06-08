/**************************************************************************
 * Copyright 2017 Tilo Wiklund, 2022 Johannes Graner
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

import scala.math.{min, max, exp, log, pow, ceil}

import org.apache.spark.rdd._
import org.apache.spark.rdd.PairRDDFunctions._
import java.math.BigInteger

import Types._

import UnfoldTreeFunctions._
import NodeLabelFunctions._
import RectangleFunctions._

// TODO: (IMPORTANT) axisAt etc should be replaced by an abstract left/right
// test function to accomodate non-axis-alinged strategies
abstract class SpatialTree extends Serializable {
  def rootCell : Rectangle

  def dimension() : Int = this.rootCell.dimension
  def volumeTotal() : Double = this.rootCell.volume

  def volumeAt(at : NodeLabel) : Double =
    this.cellAt(at).volume

  def splits: Stream[Int]

  /**
    * Axis to split along at node
    *
    * @param at
    * @return
    */
  def axisAt(at : NodeLabel) : Int

  def cellAt(at : NodeLabel) : Rectangle

  def cellAtCached() : CachedUnfoldTree[Rectangle] =
    unfoldTreeCached(rootCell)((lab, box) => box.lower(axisAt(lab.parent)),
                               (lab, box) => box.upper(axisAt(lab.parent)))

    def descendBoxPrime(point : MLVector) : Stream[(NodeLabel, Rectangle)]

  /**
    * Sub-boxes that the point belongs to at increasing depths
    *
    * @param point
    * @return
    */
  def descendBox(point : MLVector) : Stream[NodeLabel] = descendBoxPrime(point).map(_._1)
}

case class WidestSplitTree(rootCellM : Rectangle) extends SpatialTree {
  override def rootCell = rootCellM

  override def volumeAt(at : NodeLabel) : Double =
    rootCellM.volume / pow(2, at.depth)

  // NOTE: Emulates unfold, since unfold apparently only exists in Scalaz...
  override def splits : Stream[Int] = Stream.iterate((0, rootCellM.widths)) {
    case (_, ws) =>
      val i = ws.zipWithIndex.maxBy(_._1)._2
      (i, ws.updated(i, ws(i)/2))
  }.map(_._1).tail

  //TODO: Optimise
  override def axisAt(at : NodeLabel) : Int = splits(at.depth)

  override def cellAt(at : NodeLabel) : Rectangle = (at.lefts.reverse.tail zip splits).foldLeft(rootCell) {
    case (cell, (l, i)) =>
      if(l) cell.lower(i) else cell.upper(i)
  }

    // TODO: Make a more efficient implementation of this!
    // override def cellAtCached() : CachedUnfoldTree[Rectangle]

  override def descendBoxPrime(point : MLVector) : Stream[(NodeLabel, Rectangle)] =
    splits.scanLeft((rootLabel, rootCell)) {
      case ((lab, box), along) =>
        if(box.isStrictLeftOfCentre(along, point))
          (lab.left, box.lower(along))
          else
            (lab.right, box.upper(along))
    }

  /**
   * quickDescendBox - Quick splitting of single points down to given depth. Reuses arrays and uses
   *  bit operations directly on bitstring to avoid unecessary allocations.
   */
  def quickDescendBox(bits : Array[Byte], numBits : Depth, low : Array[Double], high : Array[Double], splitOrder : Array[Int],  point : MLVector, numDims : Int, depth : Depth) : NodeLabel = {

    for (i <- 0 until bits.length) {
      bits(i) = 0
    }

    rootCell.low.copyToArray(low)
    rootCell.high.copyToArray(high)
    
    var currBit = numBits - (depth + 1)
    var axis = 0
    var byte = currBit / 8 
    var shift = 7 - (currBit % 8)

    bits(byte) = bits(byte).|(1 << shift).toByte
    for (i <- 0 until depth) {
      currBit += 1
      byte = currBit / 8
      shift = 7 - (currBit % 8)
      axis = splitOrder(i)
      val mid = (low(axis) + high(axis)) / 2.0
      if (point(axis) >= mid) {
        low(axis) = mid 
        bits(byte) = bits(byte).|(1 << shift).toByte
      } else {
        high(axis) = mid 
      }
    }

    NodeLabel(new BigInt(new BigInteger(bits)))
  }

  /**
   * quickDescend - Quick splitting of points down to given depth
   */
  def quickDescend(points : Iterator[MLVector], depth : Depth) : Iterator[NodeLabel] = {

    require(depth > 0)

    /* allocation of reusable bit array representing labels with correct size from the start
     * 1 bit for sign, depth + 1 bits for representing depth
     */
    var numBits = depth + 2
    val rest = numBits % 8 
    if (rest > 0) { 
      numBits = numBits + (8 - rest)
    } 
    var bits : Array[Byte] = new Array(numBits / 8)
    val splitOrder : Array[Depth] = new Array(depth)
    val low  : Array[Double] = new Array(rootCell.dimension)
    val high  : Array[Double] = new Array(rootCell.dimension)
    var widths = rootCell.widths.toArray
    
    for (i <- 0 until depth) {
      var maxIndex = 0
      var maxVal = widths(maxIndex)
      for (j <- 1 until rootCell.dimension) {
        if (maxVal < widths(j)) {
          maxIndex = j
          maxVal = widths(j)
        }
      }
      splitOrder(i) = maxIndex 
      widths(maxIndex) = maxVal / 2
    }

    points.map(point => quickDescendBox(bits, numBits, low, high, splitOrder, point, rootCell.dimension, depth))
  }
}

  // TODO: Can we figure out some clever way to do memoisation/caching?
case class UniformSplitTree(rootCellM : Rectangle) extends SpatialTree {
  override def rootCell = rootCellM

  // def dimension() : Int = rootCell.dimension
  // def volumeTotal() : Double = rootCell.volume

  override def volumeAt(at : NodeLabel) : Double =
    rootCellM.volume / pow(2, at.depth)

  override def splits: Stream[Int] = Stream.from(1).map(_ % dimension)
  override def axisAt(at : NodeLabel) : Int =
    at.depth % dimension()

  override def cellAt(at : NodeLabel) : Rectangle =
    unfoldTree(rootCell)((lab, box) => box.lower(axisAt(lab.parent)),
                         (lab, box) => box.upper(axisAt(lab.parent)))(at)

  override def descendBoxPrime(point : MLVector) : Stream[(NodeLabel, Rectangle)] = {
    def step(lab : NodeLabel, box : Rectangle) : (NodeLabel, Rectangle) = {
      val along = axisAt(lab)

        if(box.isStrictLeftOfCentre(along, point))
          (lab.left, box.lower(along))
        else
          (lab.right, box.upper(along))
    }

    Stream.iterate((rootLabel, rootCell))(Function.tupled(step))
  }
}

object SpatialTreeFunctions {
  // def spatialTreeRootedAt(rootCell : Rectangle) : SpatialTree = SpatialTree(rootCell)
  def uniformTreeRootedAt(rootCell : Rectangle) : SpatialTree = UniformSplitTree(rootCell)
  def widestSideTreeRootedAt(rootCell : Rectangle) : WidestSplitTree = WidestSplitTree(rootCell)

  type PartitioningStrategy = RDD[MLVector] => SpatialTree

  val splitAlongWidest : PartitioningStrategy = (points => widestSideTreeRootedAt(boundingBox(points)))
  val splitUniformly : PartitioningStrategy = (points => uniformTreeRootedAt(boundingBox(points)))
}

import SpatialTreeFunctions._
