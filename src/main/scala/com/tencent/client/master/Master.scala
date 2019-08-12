package com.tencent.client.master

import io.grpc.ServerBuilder
import java.io.IOException
import java.util
import java.util.concurrent.Executors
import java.util.logging.Logger

import com.tencent.client.common.protos.AsyncModel
import org.apache.hadoop.conf.{Configurable, Configuration}

import scala.collection.JavaConversions._
import scala.collection.mutable


class Master(val port: Int, val numTask: Int, val syncModel: AsyncModel, val conf: util.Map[String, String]) {
  private val logger = Logger.getLogger(classOf[Master].getSimpleName)

  private val serverBuilder = ServerBuilder.forPort(port)
  private val masterService = new MasterService(numTask, syncModel, conf)
  private val server = serverBuilder.addService(masterService).build

  @throws[IOException]
  def start(): Unit = {
    server.start
    logger.info("Server started, listening on " + port)
    Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run(): Unit = { // Use stderr here since the logger may have been reset by its JVM shutdown hook.
        System.err.println("*** shutting down gRPC server since JVM is shutting down")
        Master.this.stop()
        System.err.println("*** server shut down")
      }
    })
  }

  def stop(): Unit = {
    if (server != null) server.shutdown
  }

  @throws[InterruptedException]
  private def blockUntilShutdown(): Unit = {
    if (server != null) server.awaitTermination()
  }
}

object Master {
  private val logger = Logger.getLogger(Master.getClass.getSimpleName)
  private var server: Master = _

  def main(args: Array[String]): Unit = {
    val conf = Map(
      "angel.output.path.deleteonexist" -> "true",
      "angel.ps.class" -> "com.tencent.angel.ps.ParameterServer",
      "angel.ps.memory.gb" -> "4",
      "angel.job.libjars" -> "",
      "angel.job.name" -> "AngelClassification-ps",
      "angel.ps.number" -> "1",
      "angel.deploy.mode" -> "LOCAL",
      "angel.am.log.level" -> "DEBUG",
      "angel.psagent.cache.sync.timeinterval.ms" -> "100000000",
      "angel.ps.heartbeat.interval.ms" -> "200",
      "angel.running.mode" -> "ANGEL_PS",
      "angel.ps.total.cores" -> "1",
      "angel.ps.cpu.vcores" -> "1",
      "angel.ps.log.level" -> "DEBUG",
      "angel.log.path" -> "file:///home/fitz/dev/sona/log",
      "angel.save.model.path" -> "file:///home/fitz/dev/sona/model",
      "plasma.store.path" -> "/home/fitz/working/arrow/plasma_store_server",
      "plasma.store.suffix" -> "/tmp/plasma",
      "plasma.store.memoryGB" -> "1",
      "python.script.name" -> "test.py",
      "python.exec.path" -> "/home/uuirs/anaconda3/envs/tf/bin/python"
    )

    start(8980, 5, AsyncModel.BSP, conf)
  }

  def embedStart(port: Int, numTask: Int, asyncModel: AsyncModel, conf: Map[String, String]): Unit = synchronized {
    if (server == null) {
      Executors.newSingleThreadExecutor().execute(new Runnable {
        override def run(): Unit = {
          start(port, numTask, asyncModel, conf)
        }
      })
    }
  }

  def embedStart(port: Int, numTask: Int, asyncModel: AsyncModel, conf: Configuration): Unit = synchronized {
    val iter =  conf.iterator()
    val confMap = new mutable.HashMap[String, String]()
    while (iter.hasNext) {
      val entry = iter.next()
      val key = entry.getKey
      if (key.contains("angel.") || key.contains("plasma.") ||
        key.contains("python.") || key.contains("ml.")) {
        confMap.put(key, entry.getValue)
      }
    }

    if (server == null) {
      Executors.newSingleThreadExecutor().execute(new Runnable {
        override def run(): Unit = {
          start(port, numTask, asyncModel, confMap.toMap)
        }
      })
    }
  }

  def embedStop(): Unit = synchronized {
    if (server != null) {
      server.stop()
    }
  }

  private def start(port: Int, numTask: Int, asyncModel: AsyncModel, conf: Map[String, String]): Unit = {
    try {
      server = new Master(port, numTask, asyncModel, conf)
      server.start()
      server.blockUntilShutdown()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        throw e
    } finally {
      if (server != null) {
        server.stop()
      }
    }

>>>>>>> 44ce6fe169f4a82630f7a846fb45648904093862
  }
}
