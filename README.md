# PyAngelPS
A python client for Angel Parameter server. You can embed the client in your distribution system, such as Spark, Flink.
The client is matser-worker architecture: 
- master: clock sync, worker monitor
- worker:
  + chief: start Angel PS (only one)
  + others: PSAgent
