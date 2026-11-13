# RTMPose per la posa, non YOLO-pose

La stima della posa non si allena: si sceglie un modello pre-addestrato. Si usa
RTMPose tramite `rtmlib` (Apache 2.0, gira con ONNX senza installare MMPose).
YOLO-pose di Ultralytics è altrettanto comodo ma è AGPL-3.0, che diventa un vincolo
se un giorno l'app è commerciale. MediaPipe è debole con più persone e con il
giocatore lontano; Apple Vision legherebbe la pipeline ai Mac.

## Consequences

- Delle persone rilevate si tengono le due più vicine alle due metà del tavolo, i cui
  angoli vengono dall'impostazione. Le altre (tavoli vicini, pubblico) si scartano.
- La scelta si verifica sulle partite pilota contando, su un campione, in quanti
  frame vicino all'impatto si vedono polso e gomito di ciascun giocatore.
