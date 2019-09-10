package com.tencent.client

import com.tencent.client.common.protos.AsyncModel
import com.tencent.client.master.Master
import com.tencent.client.worker.Worker
import org.scalatest.FunSuite

class EmbedTest extends FunSuite {
  test("EmbedTest") {
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
      "python.script.name" -> "test.py"
    )


    Master.embedStart(8980, 5, AsyncModel.BSP, conf)
    Thread.sleep(5000)
    Worker.embedStart("localhost", 8980, 9005)


    while (true) {
      Thread.sleep(1000)
    }
  }
}
