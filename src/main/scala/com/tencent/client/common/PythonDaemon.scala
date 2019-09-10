package com.tencent.client.common

import java.io.DataOutputStream
import java.util.logging.Logger

import scala.sys.process._
import scala.concurrent._
import scala.io._
import java.io._

class PythonDamon {

}

object PythonDaemon {

  val Python_Script_Name = "python.script.name"
  val Python_Exec_Path = "python.exec.path"

  private val logger = Logger.getLogger(PythonDaemon.getClass.getSimpleName)

  private var storeProcess: Process = _

  private var procOutput = new SyncVar[OutputStream]

  private val daemonModule = "pyangel.daemon"

  def startDaemon(pythonExec: String, jvmPort: Int, userScriptPath: String, plasmaName: String): Unit ={
    if (storeProcess != null)
      return
    val envVars = Map(
      "jvm_port"->jvmPort.toString,
      "user_script"->userScriptPath,
      "plasma_name"->plasmaName
    )

    val procIO = new ProcessIO(
      in => {
        procOutput.put(in)
      },
      out => {
        scala.io.Source.fromInputStream(out).getLines().foreach(println(_))
      },
      err => {
        scala.io.Source.fromInputStream(err).getLines().foreach(_.length)
      },
      true
    )

    storeProcess = Process(Seq(pythonExec, "-m", daemonModule), None, envVars.toSeq: _*).run(procIO)

  }

  def createPythonProcess(): Unit ={
    procOutput.get.write("\n".getBytes)
    procOutput.get.flush()
    println("write finish")
  }

  def close(): Unit = {
    storeProcess.destroy()
  }

  def main(args: Array[String]): Unit = {
    startDaemon("/home/uuirs/anaconda3/envs/tf/bin/python",
      9005, "/home/uuirs/dev/python_test/demo.py", "plasma_name" )
    println("start finish")
    createPythonProcess()
    createPythonProcess()
    Thread.sleep(1000)
    createPythonProcess()
  }
}
