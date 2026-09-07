import { useEffect, useRef, useState } from 'react';
import { getJob } from '../api/jobs'; import { getDocumentStatus } from '../api/document'; import type { DocumentStatusResponse, Job } from '../types/api';
export function useJobPolling(jobId:string){
 const [job,setJob]=useState<Job|null>(null); const [document,setDocument]=useState<DocumentStatusResponse|null>(null); const [loading,setLoading]=useState(true); const [networkFailures,setNetworkFailures]=useState(0); const maxProgress=useRef(0);
 useEffect(()=>{let stopped=false; let timer:number|undefined; let controller:AbortController|undefined;
  const poll=async()=>{controller=new AbortController(); try { const [j,d]=await Promise.all([getJob(jobId,controller.signal),getDocumentStatus(jobId,controller.signal)]); if(stopped)return; maxProgress.current=Math.max(maxProgress.current,j.progress); setJob({...j,progress:maxProgress.current}); setDocument(d); setNetworkFailures(0); setLoading(false); if(d.status==='READY'||j.status==='FAILED'||j.status==='CANCELLED')return; }
   catch{if(stopped)return; setNetworkFailures(v=>v+1); setLoading(false);}
   timer=window.setTimeout(poll,networkFailures>=2?4000:2000);
  }; void poll(); return()=>{stopped=true;if(timer)clearTimeout(timer);controller?.abort();};
 },[jobId,networkFailures]);
 return {job,document,loading,networkFailures};
}
