let pythonPromise;
async function python() {
  if (!pythonPromise) pythonPromise = (async () => {
    importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');
    const py = await loadPyodide({indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/'});
    const response = await fetch('/engine.py');
    if (!response.ok) throw new Error('Could not load the inspection engine.');
    await py.runPythonAsync(await response.text());
    return py;
  })().catch(error => {pythonPromise = undefined; throw error;});
  return pythonPromise;
}
self.onmessage = async ({data}) => {
  try {
    const py = await python();
    py.globals.set('payload_json', JSON.stringify(data.payload));
    let result;
    if (data.operation === 'summarize') result = await py.runPythonAsync('import json\njson.dumps(summarize(json.loads(payload_json)))');
    else {
      py.globals.set('image_bytes', new Uint8Array(data.buffer));
      result = await py.runPythonAsync('import json\np = json.loads(payload_json)\njson.dumps(analyze(image_bytes.to_py(), **p))');
      py.globals.delete('image_bytes');
    }
    py.globals.delete('payload_json');
    self.postMessage({id:data.id, result:JSON.parse(result)});
  } catch (error) { self.postMessage({id:data.id, error:'The Python engine could not run. Check your connection and try again. ' + error.message}); }
};
