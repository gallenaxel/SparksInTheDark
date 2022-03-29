package co.wiklund.disthist

import scala.reflect.ClassTag

import TruncationFunctions._
import NodeLabelFunctions._
import TruncationOperations._
import LeafMapFunctions._
import SpatialTreeFunctions._
import Types._
import org.apache.spark.mllib.linalg.Vectors

object DropHead extends Enumeration {
  type DropHead = Value
  val L1, L2, Both = Value
}
import DropHead._

object TruncationOperations {
  /**
    * The leaves of the tree resulting from `RPUnion(t1, t2)`
    */
  def rpUnion(t1: Truncation, t2: Truncation): Truncation = {

    def getDrop(n1: NodeLabel, n2: NodeLabel): (Vector[NodeLabel], DropHead) = {
      if (n1 == n2)
        (Vector(n1), Both)
      else if (isAncestorOf(n2, n1))
        ((n1 +: n1.sibling +: n1.ancestors.take(n1.depth - n2.depth - 1).map(_.sibling).toVector).sorted(leftRightOrd), Both)
      else if (isAncestorOf(n1, n2))
        ((n2 +: n2.sibling +: n2.ancestors.take(n2.depth - n1.depth - 1).map(_.sibling).toVector).sorted(leftRightOrd), Both)
      else if (isLeftOf(n1, n2))
        (Vector(n1), L1)
      else
        (Vector(n2), L2)
    }

    @scala.annotation.tailrec
    def unionHelper(
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
          acc ++ next.filter( !acc.contains(_) ), 
          if (drop != L2) l1.tail else l1,
          if (drop != L1) l2.tail else l2
        )
      }
    }

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

  def mrpOperate[A:ClassTag](leafMap1: LeafMap[A], leafMap2: LeafMap[A], op: (A, A) => A, base: A): LeafMap[A] = {

    val unionLeaves = rpUnion(leafMap1.truncation, leafMap2.truncation).leaves

    val operatedVals = unionLeaves.foldLeft((Vector.empty[A], leafMap1.leaves zip leafMap1.vals, leafMap2.leaves zip leafMap2.vals)){ case ((acc, l1, l2), newNode) => 
      val v1OptIndex = l1.indexWhere{ case(node, _) => node == newNode || isAncestorOf(node, newNode) }
      val v2OptIndex = l2.indexWhere{ case(node, _) => node == newNode || isAncestorOf(node, newNode) }

      val newv = (v1OptIndex, v2OptIndex) match {
        case (i, -1) => op(l1(i)._2, base)
        case (-1, i) => op(base, l2(i)._2)
        case (i, k) => op(l1(i)._2, l2(k)._2)
        case _ => throw new IllegalArgumentException("should not happen")
      }

      def newIter1 = v1OptIndex match {
        case -1 => l1
        case i => if (l1(i)._1 == newNode) l1.drop(i+1) else l1
      }

      def newIter2 = v2OptIndex match {
        case -1 => l2
        case i => if (l2(i)._1 == newNode) l2.drop(i+1) else l2
      }

      ( acc :+ newv, 
        if (l1.isEmpty) l1 else newIter1, 
        if (l2.isEmpty) l2 else newIter2 
      )
    }._1

    LeafMap(Truncation(unionLeaves), operatedVals)
  }
}