import { apiJson } from './client'; import type { Job } from '../types/api';
export const createJob=(source_url:string,signal?:AbortSignal)=>apiJson<Job>('/api/jobs',{method:'POST',body:JSON.stringify({source_url}),signal});
export const getJob=(id:string,signal?:AbortSignal)=>apiJson<Job>(`/api/jobs/${encodeURIComponent(id)}`,{signal});
export const cancelJob=(id:string)=>apiJson<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`,{method:'POST'});
export const retryJob=(id:string)=>apiJson<Job>(`/api/jobs/${encodeURIComponent(id)}/retry`,{method:'POST'});
