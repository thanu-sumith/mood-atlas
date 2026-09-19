const text=document.querySelector('#text'),button=document.querySelector('#submit'),status=document.querySelector('#status'),scores=document.querySelector('#scores');
text.addEventListener('input',()=>document.querySelector('#count').textContent=`${text.value.length} / 3000`);
document.querySelector('#sample').addEventListener('click',()=>{text.value='I am so happy we won the cricket match! I am excited about the next game.';text.dispatchEvent(new Event('input'));text.focus();});
document.querySelector('#analyze').addEventListener('submit',async event=>{
 event.preventDefault();button.disabled=true;status.textContent='Reading emotion patterns…';scores.replaceChildren();
 try{
  const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.value}),signal:AbortSignal.timeout(30000)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'The model is unavailable. Please try again.');
  status.textContent=result.summary;
  result.emotions.forEach(item=>{const row=document.createElement('div');row.className='score';row.textContent=item.label;scores.append(row);});
  result.cues.forEach(item=>{const row=document.createElement('div');row.className='score';const title=document.createElement('strong'),note=document.createElement('p');title.textContent=item.label+' · explicit wording';note.textContent=item.note;row.append(title,note);scores.append(row);});
 }catch(error){status.textContent=error.name==='TimeoutError'?'The request timed out. Please try again.':error.message;}finally{button.disabled=false;}
});
