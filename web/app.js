const $ = (id) => document.getElementById(id);

function filas(tabla, pares) {
  tabla.querySelector("tbody").innerHTML = pares
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

async function refrescar() {
  try {
    const r = await fetch("/health", { cache: "no-store" });
    const d = await r.json();

    const estado = $("estado");
    estado.textContent = d.corriendo ? "en directo" : "detenido";
    estado.className = "pill " + (d.corriendo ? "pill-ok" : "pill-espera");

    $("fps").textContent = (d.fps ?? 0).toFixed(1);
    $("dets").textContent = d.detecciones ? d.detecciones.length : 0;

    const t = d.tiempos_ms || {};
    $("latencia").textContent = t.inferencia ? t.inferencia.p50.toFixed(2) : "—";

    const orden = ["decode", "preproceso", "inferencia", "postproceso", "anotado", "encode"];
    filas($("tiempos"), orden.filter((k) => t[k]).map((k) => [k, `${t[k].p50.toFixed(2)} / ${t[k].p95.toFixed(2)}`]));

    filas($("frames"), [
      ["leídos", d.frames_leidos],
      ["procesados", d.frames_procesados],
      ["descartados", d.frames_descartados],
    ]);

    filas($("config"), [
      ["engine", d.engine],
      ["vídeo", d.video],
      ["resolución", (d.resolucion || []).join(" × ")],
      ["fps del vídeo", d.fps_video],
    ]);
  } catch (e) {
    const estado = $("estado");
    estado.textContent = "sin conexión";
    estado.className = "pill pill-espera";
  }
}

refrescar();
setInterval(refrescar, 1000);
