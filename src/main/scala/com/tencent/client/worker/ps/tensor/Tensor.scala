package com.tencent.client.worker.ps.tensor


import java.util.logging.Logger

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.matrix._
import com.tencent.angel.ml.math2.vector._
import com.tencent.client.common.Utils
import com.tencent.client.worker.ps.variable.{Initializer, NormalInitializer}


class Tensor(name: String, dim: Int, shape: Array[Long], dtype: String, validIndexNum: Long,
             initializer: Initializer = new NormalInitializer(0.0, 1e-6))
  extends TensorLike(name, dim, shape, dtype, validIndexNum, initializer) {
  private val logger = Logger.getLogger(classOf[Tensor].getClass.getSimpleName)
  override protected val meta: TensorMeta = new TensorMeta(name, dtype, dim, shape, validIndexNum)

  override def getMeta: TensorMeta = meta

  protected def doPull(epoch: Int, idxs: Matrix): Matrix = {
    var indices: Vector = null
    if (idxs != null) {
      indices = idxs.getRow(0)
    }
    val rowIds = (0 until meta.getMatrixContext.getRowNum).toArray

    val pulled = if (indices != null) {
      indices match {
        case v: IntIntVector if v.isDense =>
          matClient.clock()
          matClient.get(rowIds, v.getStorage.getValues)
        case v: IntDummyVector =>
          matClient.get(rowIds, v.getIndices)
        case v: IntLongVector if v.isDense =>
          matClient.get(rowIds, v.getStorage.getValues)
        case v: LongDummyVector =>
          matClient.get(rowIds, v.getIndices)
      }
    } else {
      matClient.getRows(rowIds, true)
    }
    logger.info("pulled dopull: "+pulled.map(_.std()).mkString(":"))

    Utils.vectorArray2Matrix(pulled)
  }

  protected def doPush(data: Matrix, alpha: Double): Unit = {
    val matrixId = matClient.getMatrixId

    data match {
      case dblas: BlasDoubleMatrix =>
        val row = VFactory.denseDoubleVector(dblas.getData)
        row.imul(alpha)
        row.setMatrixId(matrixId)
        row.setRowId(0)
        matClient.update(row)
      case fblas: BlasFloatMatrix =>
        val row = VFactory.denseFloatVector(fblas.getData)
        row.imul(alpha)
        row.setMatrixId(matrixId)
        row.setRowId(0)
        matClient.update(row)
      case rbase: RowBasedMatrix[_] =>
        val rowIds = (0 until meta.getMatrixContext.getRowNum).toArray
        val rows = rowIds.map { rowId =>
          val row = rbase.getRow(rowId)
          row.imul(alpha)
          row.setMatrixId(matrixId)
          row.setRowId(rowId)

          row
        }

        matClient.update(rowIds, rows.asInstanceOf[Array[Vector]])
    }
  }

}
