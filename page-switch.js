'use strict';
document.getElementById('pageSwitch').addEventListener('change',event=>{const pages={regression:'index.html',selection:'selection.html',upload:'upload.html'};const url=pages[event.target.value];if(url)window.location.assign(url);});
