importScripts('regression-engine.js');
self.onmessage=event=>{try{self.postMessage({result:RegressionEngine.analyze(event.data.data,event.data.config)});}catch(e){self.postMessage({error:e.message});}};
