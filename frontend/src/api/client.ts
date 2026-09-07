import type { ApiError } from '../types/api';
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
export async function apiJson<T>(path:string, options:RequestInit={}):Promise<T>{
  let response:Response;
  try { response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers:{ 'Content-Type':'application/json', ...(options.headers||{}) } }); }
  catch { throw { code:'network_error', message:'Cannot connect to Notify backend.', status:0 } satisfies ApiError; }
  if(!response.ok){
    let payload:unknown; try { payload=await response.json(); } catch { payload=null; }
    const obj = payload && typeof payload==='object' ? payload as Record<string,unknown> : {};
    const detail = obj.detail && typeof obj.detail==='object' ? obj.detail as Record<string,unknown> : {};
    throw { code:String(detail.code||obj.error||`http_${response.status}`), message:String(detail.message||obj.message||'Request failed.'), status:response.status } satisfies ApiError;
  }
  return response.json() as Promise<T>;
}
export function isApiError(value:unknown):value is ApiError { return !!value && typeof value==='object' && 'code' in value && 'status' in value; }
